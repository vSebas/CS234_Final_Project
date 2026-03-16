#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"

data_dir="${DATA_DIR:-data/datasets}"
context_length="${CONTEXT_LENGTH:-30}"
num_workers="${NUM_WORKERS:-4}"
device="${DEVICE:-cuda}"
lambda_x="${LAMBDA_X:-}"
repair_weight="${REPAIR_WEIGHT:-}"
shift_fraction="${SHIFT_FRACTION:-}"
repair_fraction="${REPAIR_FRACTION:-}"
hard_repair_fraction="${HARD_REPAIR_FRACTION:-}"
postproj_repair_fraction="${POSTPROJ_REPAIR_FRACTION:-}"
python_bin="${PYTHON_BIN:-python}"
amp="${AMP:-1}"
resume="${RESUME:-}"
n_layer="${N_LAYER:-}"
n_head="${N_HEAD:-}"
d_model="${D_MODEL:-}"
dropout="${DROPOUT:-}"
early_stop_patience="${EARLY_STOP_PATIENCE:-}"
learning_rate="${LEARNING_RATE:-}"
save_every="${SAVE_EVERY:-}"

case "${mode}" in
  smoke)
    output_dir="${OUTPUT_DIR:-dt/checkpoints/smoke_full_root_gpu}"
    batch_size="${BATCH_SIZE:-16}"
    num_epochs="${NUM_EPOCHS:-1}"
    if [[ -z "${lambda_x}" ]]; then
      lambda_x="0.5"
    fi
    ;;
  full)
    output_dir="${OUTPUT_DIR:-dt/checkpoints/full_run1}"
    batch_size="${BATCH_SIZE:-64}"
    num_epochs="${NUM_EPOCHS:-10}"
    if [[ -z "${lambda_x}" ]]; then
      lambda_x="0.5"
    fi
    ;;
  full_action)
    output_dir="${OUTPUT_DIR:-dt/checkpoints/full_run_lambda0}"
    batch_size="${BATCH_SIZE:-64}"
    num_epochs="${NUM_EPOCHS:-40}"
    if [[ -z "${lambda_x}" ]]; then
      lambda_x="0.0"
    fi
    ;;
  full_action_hard)
    output_dir="${OUTPUT_DIR:-dt/checkpoints/full_run_lambda0_hard}"
    batch_size="${BATCH_SIZE:-64}"
    num_epochs="${NUM_EPOCHS:-40}"
    if [[ -z "${lambda_x}" ]]; then
      lambda_x="0.0"
    fi
    if [[ -z "${shift_fraction}" ]]; then
      shift_fraction="0.75"
    fi
    if [[ -z "${repair_fraction}" ]]; then
      repair_fraction="0.10"
    fi
    if [[ -z "${hard_repair_fraction}" ]]; then
      hard_repair_fraction="0.15"
    fi
    ;;
  full_action_postproj)
    output_dir="${OUTPUT_DIR:-dt/checkpoints/full_run_lambda0_postproj}"
    batch_size="${BATCH_SIZE:-64}"
    num_epochs="${NUM_EPOCHS:-40}"
    if [[ -z "${lambda_x}" ]]; then
      lambda_x="0.0"
    fi
    if [[ -z "${shift_fraction}" ]]; then
      shift_fraction="0.85"
    fi
    if [[ -z "${repair_fraction}" ]]; then
      repair_fraction="0.10"
    fi
    if [[ -z "${hard_repair_fraction}" ]]; then
      hard_repair_fraction="0.00"
    fi
    if [[ -z "${postproj_repair_fraction}" ]]; then
      postproj_repair_fraction="0.05"
    fi
    ;;
  oval_fatrop_clean)
    output_dir="${OUTPUT_DIR:-dt/checkpoints/oval_fatrop_clean_train20}"
    batch_size="${BATCH_SIZE:-128}"
    num_epochs="${NUM_EPOCHS:-20}"
    data_dir="${DATA_DIR:-data/datasets/Oval_Track_260m_shifts_fatrop_clean,data/datasets/Oval_Track_260m_repairs_hard_fatrop_clean}"
    num_workers="${NUM_WORKERS:-4}"
    device="${DEVICE:-cuda}"
    if [[ -z "${lambda_x}" ]]; then
      lambda_x="0.0"
    fi
    ;;
  oval_fatrop_improved)
    output_dir="${OUTPUT_DIR:-dt/checkpoints/oval_fatrop_improved_ctx50_m6x192}"
    batch_size="${BATCH_SIZE:-128}"
    num_epochs="${NUM_EPOCHS:-20}"
    data_dir="${DATA_DIR:-data/datasets/Oval_Track_260m_shifts_fatrop_clean,data/datasets/Oval_Track_260m_repairs_hard_fatrop_clean}"
    context_length="${CONTEXT_LENGTH:-50}"
    num_workers="${NUM_WORKERS:-0}"
    device="${DEVICE:-cuda}"
    if [[ -z "${lambda_x}" ]]; then
      lambda_x="0.1"
    fi
    if [[ -z "${shift_fraction}" ]]; then
      shift_fraction="0.995"
    fi
    if [[ -z "${repair_fraction}" ]]; then
      repair_fraction="0.00"
    fi
    if [[ -z "${hard_repair_fraction}" ]]; then
      hard_repair_fraction="0.005"
    fi
    if [[ -z "${postproj_repair_fraction}" ]]; then
      postproj_repair_fraction="0.00"
    fi
    if [[ -z "${n_layer}" ]]; then
      n_layer="6"
    fi
    if [[ -z "${n_head}" ]]; then
      n_head="4"
    fi
    if [[ -z "${d_model}" ]]; then
      d_model="192"
    fi
    if [[ -z "${dropout}" ]]; then
      dropout="0.1"
    fi
    if [[ -z "${early_stop_patience}" ]]; then
      early_stop_patience="3"
    fi
    ;;
  oval_fatrop_improved_postproj)
    output_dir="${OUTPUT_DIR:-dt/checkpoints/oval_fatrop_improved_postproj_ft}"
    batch_size="${BATCH_SIZE:-128}"
    num_epochs="${NUM_EPOCHS:-20}"
    data_dir="${DATA_DIR:-data/datasets/Oval_Track_260m_shifts_fatrop_clean,data/datasets/Oval_Track_260m_repairs_hard_fatrop_clean,data/datasets/Oval_Track_260m_repairs_postproj_fatrop_clean}"
    context_length="${CONTEXT_LENGTH:-50}"
    num_workers="${NUM_WORKERS:-0}"
    device="${DEVICE:-cuda}"
    if [[ -z "${lambda_x}" ]]; then
      lambda_x="0.1"
    fi
    if [[ -z "${shift_fraction}" ]]; then
      shift_fraction="0.90"
    fi
    if [[ -z "${repair_fraction}" ]]; then
      repair_fraction="0.00"
    fi
    if [[ -z "${hard_repair_fraction}" ]]; then
      hard_repair_fraction="0.00"
    fi
    if [[ -z "${postproj_repair_fraction}" ]]; then
      postproj_repair_fraction="0.10"
    fi
    if [[ -z "${n_layer}" ]]; then
      n_layer="6"
    fi
    if [[ -z "${n_head}" ]]; then
      n_head="4"
    fi
    if [[ -z "${d_model}" ]]; then
      d_model="192"
    fi
    if [[ -z "${dropout}" ]]; then
      dropout="0.1"
    fi
    if [[ -z "${early_stop_patience}" ]]; then
      early_stop_patience="3"
    fi
    if [[ -z "${resume}" ]]; then
      resume="dt/checkpoints/oval_fatrop_improved_ctx50_m6x192/checkpoints/checkpoint_best.pt"
      if [[ ! -f "${resume}" ]]; then
        echo "[dt/run_train] warning: default resume checkpoint missing (${resume}); falling back to auto"
        resume="auto"
      fi
    fi
    ;;
  oval_fatrop_improved_postproj_ft2)
    output_dir="${OUTPUT_DIR:-dt/checkpoints/oval_fatrop_improved_postproj_ft2}"
    batch_size="${BATCH_SIZE:-128}"
    num_epochs="${NUM_EPOCHS:-15}"
    data_dir="${DATA_DIR:-data/datasets/Oval_Track_260m_shifts_fatrop_clean,data/datasets/Oval_Track_260m_repairs_hard_fatrop_clean,data/datasets/Oval_Track_260m_repairs_postproj_fatrop_clean}"
    context_length="${CONTEXT_LENGTH:-50}"
    num_workers="${NUM_WORKERS:-0}"
    device="${DEVICE:-cuda}"
    if [[ -z "${lambda_x}" ]]; then
      lambda_x="0.1"
    fi
    if [[ -z "${learning_rate}" ]]; then
      learning_rate="5e-5"
    fi
    if [[ -z "${shift_fraction}" ]]; then
      shift_fraction="0.80"
    fi
    if [[ -z "${repair_fraction}" ]]; then
      repair_fraction="0.00"
    fi
    if [[ -z "${hard_repair_fraction}" ]]; then
      hard_repair_fraction="0.05"
    fi
    if [[ -z "${postproj_repair_fraction}" ]]; then
      postproj_repair_fraction="0.15"
    fi
    if [[ -z "${n_layer}" ]]; then
      n_layer="6"
    fi
    if [[ -z "${n_head}" ]]; then
      n_head="4"
    fi
    if [[ -z "${d_model}" ]]; then
      d_model="192"
    fi
    if [[ -z "${dropout}" ]]; then
      dropout="0.1"
    fi
    if [[ -z "${early_stop_patience}" ]]; then
      early_stop_patience="3"
    fi
    if [[ -z "${save_every}" ]]; then
      save_every="1"
    fi
    if [[ -z "${resume}" ]]; then
      resume="dt/checkpoints/oval_fatrop_improved_ctx50_m6x192/checkpoints/checkpoint_best.pt"
      if [[ ! -f "${resume}" ]]; then
        echo "[dt/run_train] warning: default resume checkpoint missing (${resume}); falling back to auto"
        resume="auto"
      fi
    fi
    ;;
  *)
    echo "Usage: $0 [smoke|full|full_action|full_action_hard|full_action_postproj|oval_fatrop_clean|oval_fatrop_improved|oval_fatrop_improved_postproj|oval_fatrop_improved_postproj_ft2]"
    echo "Optional env overrides: OUTPUT_DIR, DATA_DIR, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, CONTEXT_LENGTH, DEVICE, LAMBDA_X, REPAIR_WEIGHT, SHIFT_FRACTION, REPAIR_FRACTION, HARD_REPAIR_FRACTION, POSTPROJ_REPAIR_FRACTION, N_LAYER, N_HEAD, D_MODEL, DROPOUT, EARLY_STOP_PATIENCE, LEARNING_RATE, SAVE_EVERY"
    exit 1
    ;;
esac

if [[ -z "${repair_weight}" ]]; then
  repair_weight="1.0"
fi
if [[ -z "${resume}" ]]; then
  resume="auto"
fi

mkdir -p "${output_dir}"

echo "[dt/run_train] mode=${mode}"
echo "[dt/run_train] data_dir=${data_dir}"
echo "[dt/run_train] output_dir=${output_dir}"
echo "[dt/run_train] batch_size=${batch_size}"
echo "[dt/run_train] num_epochs=${num_epochs}"
echo "[dt/run_train] num_workers=${num_workers}"
echo "[dt/run_train] context_length=${context_length}"
echo "[dt/run_train] device=${device}"
echo "[dt/run_train] python_bin=${python_bin}"
echo "[dt/run_train] amp=${amp}"
echo "[dt/run_train] lambda_x=${lambda_x}"
echo "[dt/run_train] repair_weight=${repair_weight}"
if [[ -n "${n_layer}" || -n "${n_head}" || -n "${d_model}" || -n "${dropout}" ]]; then
  echo "[dt/run_train] n_layer=${n_layer:-default}"
  echo "[dt/run_train] n_head=${n_head:-default}"
  echo "[dt/run_train] d_model=${d_model:-default}"
  echo "[dt/run_train] dropout=${dropout:-default}"
fi
if [[ -n "${early_stop_patience}" ]]; then
  echo "[dt/run_train] early_stop_patience=${early_stop_patience}"
fi
if [[ -n "${learning_rate}" ]]; then
  echo "[dt/run_train] learning_rate=${learning_rate}"
fi
if [[ -n "${save_every}" ]]; then
  echo "[dt/run_train] save_every=${save_every}"
fi
if [[ -n "${shift_fraction}" || -n "${repair_fraction}" || -n "${hard_repair_fraction}" || -n "${postproj_repair_fraction}" ]]; then
  echo "[dt/run_train] shift_fraction=${shift_fraction:-unset}"
  echo "[dt/run_train] repair_fraction=${repair_fraction:-unset}"
  echo "[dt/run_train] hard_repair_fraction=${hard_repair_fraction:-unset}"
  echo "[dt/run_train] postproj_repair_fraction=${postproj_repair_fraction:-unset}"
fi
echo "[dt/run_train] resume=${resume}"

cmd=(
  "${python_bin}" -u dt/train.py
  --data-dir "${data_dir}"
  --output-dir "${output_dir}"
  --context-length "${context_length}"
  --batch-size "${batch_size}"
  --num-epochs "${num_epochs}"
  --num-workers "${num_workers}"
  --device "${device}"
  --amp "${amp}"
  --lambda-x "${lambda_x}"
  --repair-weight "${repair_weight}"
  --resume "${resume}"
)

if [[ -n "${learning_rate}" ]]; then
  cmd+=(--lr "${learning_rate}")
fi
if [[ -n "${save_every}" ]]; then
  cmd+=(--save-every "${save_every}")
fi

if [[ -n "${n_layer}" ]]; then
  cmd+=(--n-layer "${n_layer}")
fi
if [[ -n "${n_head}" ]]; then
  cmd+=(--n-head "${n_head}")
fi
if [[ -n "${d_model}" ]]; then
  cmd+=(--d-model "${d_model}")
fi
if [[ -n "${dropout}" ]]; then
  cmd+=(--dropout "${dropout}")
fi
if [[ -n "${early_stop_patience}" ]]; then
  cmd+=(--early-stop-patience "${early_stop_patience}")
fi

if [[ -n "${shift_fraction}" || -n "${repair_fraction}" || -n "${hard_repair_fraction}" || -n "${postproj_repair_fraction}" ]]; then
  cmd+=(
    --shift-fraction "${shift_fraction}"
    --repair-fraction "${repair_fraction}"
    --hard-repair-fraction "${hard_repair_fraction}"
  )
  if [[ -n "${postproj_repair_fraction}" ]]; then
    cmd+=(--postproj-repair-fraction "${postproj_repair_fraction}")
  fi
fi

"${cmd[@]}"
