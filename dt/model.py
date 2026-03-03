"""
Decision Transformer model for trajectory optimization warm-starting.

Architecture based on PLAN.md Section 3:
- Causal GPT-style transformer
- Modality embeddings: RTG, state_aug, action
- Two-head outputs: action prediction + state prediction
- Context length K=30

Hyperparameters (baseline):
- layers: 4
- heads: 4
- embedding dim (d_model): 128
- MLP hidden dim: 512
- dropout: 0.1
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DTConfig:
    """Configuration for Decision Transformer."""
    # State/action dimensions (derived from problem)
    state_dim: int = 8        # [ux, uy, r, e, dpsi, pos_E, pos_N, yaw_world]
    track_dim: int = 2        # [kappa, half_width]
    obstacle_slots: int = 8   # M = number of obstacle slots
    obstacle_feat_dim: int = 3  # [Δs, e_obs - e_k, r_obs]
    act_dim: int = 2          # [delta, Fx]

    # Architecture
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int = 512           # MLP hidden dim (4 * d_model)
    dropout: float = 0.1

    # Context
    context_length: int = 30  # K
    max_ep_len: int = 200     # Maximum episode length for timestep embedding

    # Training
    action_tanh: bool = False  # No tanh on actions (use explicit clipping)

    @property
    def state_aug_dim(self) -> int:
        """Total augmented state dimension (vehicle + track + obstacles)."""
        return self.state_dim + self.track_dim + self.obstacle_slots * self.obstacle_feat_dim


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: DTConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0

        self.n_head = config.n_head
        self.d_head = config.d_model // config.n_head
        self.d_model = config.d_model

        # Key, query, value projections
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.size()

        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))

        # Causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(causal_mask == 0, float('-inf'))

        query_mask = None

        # Apply attention mask if provided
        if attention_mask is not None:
            # Key mask: (B, T) -> (B, 1, 1, T)
            key_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            att = att.masked_fill(key_mask == 0, float('-inf'))

            # Query mask: (B, T) -> (B, 1, T, 1)
            # Invalid padded queries can otherwise end up with an all -inf row,
            # which makes softmax produce NaNs.
            query_mask = attention_mask.unsqueeze(1).unsqueeze(3)
            att = torch.where(query_mask == 0, torch.zeros_like(att), att)

        att = F.softmax(att, dim=-1)

        if query_mask is not None:
            att = torch.where(query_mask == 0, torch.zeros_like(att), att)

        att = self.attn_dropout(att)

        # Apply attention to values
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        if attention_mask is not None:
            y = y * attention_mask.unsqueeze(-1).to(y.dtype)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: DTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff)
        self.c_proj = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-layer normalization."""

    def __init__(self, config: DTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for trajectory optimization warm-starting.

    Tokenization: (RTG_k, state_aug_k, action_k) for each timestep k.

    State augmentation includes:
    - Vehicle/path observation: [ux, uy, r, e, dpsi, pos_E, pos_N, yaw_world]
    - Track features: [kappa, half_width]
    - Obstacle features: M * [Δs, e_obs - e_k, r_obs] (sorted by Δs, padded)

    Outputs:
    - Action prediction: [delta, Fx]
    - State prediction: next observation (for auxiliary loss)
    """

    def __init__(self, config: DTConfig):
        super().__init__()
        self.config = config

        # Embeddings for each modality
        self.embed_timestep = nn.Embedding(config.max_ep_len, config.d_model)
        self.embed_return = nn.Linear(1, config.d_model)
        self.embed_state = nn.Linear(config.state_aug_dim, config.d_model)
        self.embed_action = nn.Linear(config.act_dim, config.d_model)

        # Layer normalization after embedding
        self.embed_ln = nn.LayerNorm(config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Output layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Prediction heads
        # Action head: predict action given state (from state position in sequence)
        self.predict_action = nn.Linear(config.d_model, config.act_dim)

        # State head: predict next state observation (from action position in sequence)
        self.predict_state = nn.Linear(config.d_model, config.state_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,       # (B, K, state_aug_dim)
        actions: torch.Tensor,      # (B, K, act_dim)
        returns_to_go: torch.Tensor,  # (B, K, 1)
        timesteps: torch.Tensor,    # (B, K) or (B, K, 1)
        attention_mask: Optional[torch.Tensor] = None  # (B, K)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Decision Transformer.

        Args:
            states: Augmented state observations
            actions: Control actions
            returns_to_go: RTG values
            timesteps: Timestep indices
            attention_mask: Mask for valid tokens (1 = attend, 0 = ignore)

        Returns:
            action_preds: Predicted actions (B, K, act_dim)
            state_preds: Predicted next states (B, K, state_dim)
        """
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # Handle timesteps shape
        if timesteps.dim() == 3:
            timesteps = timesteps.squeeze(-1)
        timesteps = timesteps.long()

        # Embed each modality
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Add time embeddings (positional encoding)
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Stack as (RTG, state, action) sequence
        # This makes: (R_0, s_0, a_0, R_1, s_1, a_1, ...)
        # Shape: (B, K, 3, d_model) -> (B, 3*K, d_model)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=2
        ).reshape(batch_size, 3 * seq_length, self.config.d_model)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # Expand attention mask for stacked sequence
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=2
        ).reshape(batch_size, 3 * seq_length)

        # Pass through transformer blocks
        x = stacked_inputs
        for block in self.blocks:
            x = block(x, stacked_attention_mask)

        x = self.ln_f(x)

        # Reshape back to (B, K, 3, d_model)
        x = x.reshape(batch_size, seq_length, 3, self.config.d_model)

        # Extract predictions:
        # - Action: predicted from state token (index 1)
        # - Next state: predicted from action token (index 2)
        action_preds = self.predict_action(x[:, :, 1, :])  # (B, K, act_dim)
        state_preds = self.predict_state(x[:, :, 2, :])    # (B, K, state_dim)

        return action_preds, state_preds

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get action prediction for the last timestep (inference mode).

        Args:
            states: (1, T, state_aug_dim) or (T, state_aug_dim)
            actions: (1, T, act_dim) or (T, act_dim)
            returns_to_go: (1, T, 1) or (T, 1)
            timesteps: (1, T) or (T,)

        Returns:
            action: (act_dim,) predicted action for the last timestep
        """
        # Ensure batch dimension
        if states.dim() == 2:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            returns_to_go = returns_to_go.unsqueeze(0) if returns_to_go.dim() == 2 else returns_to_go.unsqueeze(0).unsqueeze(-1)
            timesteps = timesteps.unsqueeze(0)

        # Truncate to context length
        K = self.config.context_length
        if states.shape[1] > K:
            states = states[:, -K:]
            actions = actions[:, -K:]
            returns_to_go = returns_to_go[:, -K:]
            timesteps = timesteps[:, -K:]

        # Pad if shorter than context length
        T = states.shape[1]
        if T < K:
            pad_len = K - T
            device = states.device

            attention_mask = torch.cat([
                torch.zeros(1, pad_len, device=device),
                torch.ones(1, T, device=device)
            ], dim=1).long()

            states = torch.cat([
                torch.zeros(1, pad_len, self.config.state_aug_dim, device=device),
                states
            ], dim=1)
            actions = torch.cat([
                torch.zeros(1, pad_len, self.config.act_dim, device=device),
                actions
            ], dim=1)
            returns_to_go = torch.cat([
                torch.zeros(1, pad_len, 1, device=device),
                returns_to_go
            ], dim=1)
            timesteps = torch.cat([
                torch.zeros(1, pad_len, device=device).long(),
                timesteps
            ], dim=1)
        else:
            attention_mask = None

        action_preds, _ = self.forward(
            states, actions, returns_to_go, timesteps, attention_mask
        )

        return action_preds[0, -1]  # Return last action prediction


def build_model(
    state_dim: int = 8,
    act_dim: int = 2,
    track_dim: int = 2,
    obstacle_slots: int = 8,
    n_layer: int = 4,
    n_head: int = 4,
    d_model: int = 128,
    context_length: int = 30,
    dropout: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> DecisionTransformer:
    """Build and return a Decision Transformer model."""
    config = DTConfig(
        state_dim=state_dim,
        track_dim=track_dim,
        obstacle_slots=obstacle_slots,
        act_dim=act_dim,
        n_layer=n_layer,
        n_head=n_head,
        d_model=d_model,
        context_length=context_length,
        dropout=dropout,
    )
    model = DecisionTransformer(config)
    return model.to(device)
