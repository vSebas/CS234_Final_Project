import dataclasses
import pickle
from dataclasses import InitVar, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from casadi_tools import math
from matplotlib.axes import Axes
from python_data_parsers import units
from python_data_parsers.units import SI_PREFIX
from scipy import io, spatial
from typing_extensions import Self

_NUM_KD_MATCHES = 100
_S_WINDOW = 20
np.ndarray = np.ndarray


def _check_array_sizes(arrays: list[np.ndarray]):
    for array in arrays:
        if array.ndim > 1:
            raise ValueError("All inputs must be row or column vectors")

    for arr1, arr2 in pairwise(arrays):
        if not (arr1.size == arr2.size):
            raise ValueError("All inputs must be the same size")


def _format_arrays(arrays: list[np.ndarray]) -> list[np.ndarray]:
    return [np.atleast_1d(np.array(array).squeeze()) for array in arrays]


def _cast_array_to_float(array: np.ndarray) -> np.ndarray | float:
    try:
        return array.squeeze().item()
    except ValueError:
        return array.squeeze()


def _cast_array_list_to_floats(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple([_cast_array_to_float(array) for array in arrays])


def _project_veh_on_path(
    rel_east: np.ndarray, rel_north: np.ndarray, path_psi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    path_tangent = np.row_stack((-np.sin(path_psi), np.cos(path_psi)))
    path_normal = np.array([[0, -1], [1, 0]]) @ path_tangent

    del_pos = np.column_stack((rel_east, rel_north))
    dot_s = np.diag(del_pos @ path_tangent)
    dot_e = np.diag(del_pos @ path_normal)

    proj_s = dot_s / np.linalg.norm(path_tangent, axis=0, keepdims=True)
    proj_e = dot_e / np.linalg.norm(path_normal, axis=0, keepdims=True)

    return proj_s.squeeze(), proj_e.squeeze()


@dataclass(eq=False)
class SimpleWorld:
    s_m: np.ndarray
    
    east_m: np.ndarray
    north_m: np.ndarray
    up_m: np.ndarray

    psi_rad: np.ndarray
    k_1pm: np.ndarray

    inner_bound_east_m: np.ndarray
    inner_bound_north_m: np.ndarray
    inner_bound_up_m: np.ndarray

    outer_bound_east_m: np.ndarray
    outer_bound_north_m: np.ndarray
    outer_bound_up_m: np.ndarray

    ux_des_mps: np.ndarray
    e_des_m: np.ndarray

    track_width_m: np.ndarray

    _tree: spatial.KDTree = field(init=False, repr=False)

    @property
    def size(self) -> int:
        return self.s_m.size

    @property
    def length_m(self) -> float:
        return self.s_m[-1]

    @property
    def radius_m(self) -> np.ndarray:
        return 1.0 / self.k_1pm

    def head(self) -> Self:
        wdict = {key: val[:1] for key, val in self.asdict().items()}
        return type(self)(**wdict)

    def tail(self) -> Self:
        wdict = {key: val[-1:] for key, val in self.asdict().items()}
        return type(self)(**wdict)

    def enu_to_seu(
        self, east_m: np.ndarray, north_m: np.ndarray, psi_rad: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        east_m_1d = np.array(east_m).squeeze()
        north_m_1d = np.array(north_m).squeeze()
        psi_rad_1d = np.array(psi_rad).squeeze()

        if (east_m_1d.ndim > 1) or (north_m_1d.ndim > 1) or (psi_rad_1d.ndim > 1):
            raise ValueError("All inputs must be row or column vectors")

        if not (east_m_1d.size == north_m_1d.size == psi_rad_1d.size):
            raise ValueError("All inputs must be the same size")

        query_array = np.hstack((east_m_1d.reshape(-1, 1), north_m_1d.reshape(-1, 1)))
        _, idx = self._tree.query(query_array)

        idx = idx.reshape(1, -1)

        matched_psi = np.array(self.psi_rad)[idx]
        matched_E = np.array(self.east_m)[idx]
        matched_N = np.array(self.north_m)[idx]
        matched_s = np.array(self.s_m)[idx].squeeze()

        path_pos = np.vstack((matched_E, matched_N))
        veh_pos_from_path = query_array.T - path_pos

        path_tangent = np.vstack((-np.sin(matched_psi), np.cos(matched_psi)))
        path_normal = np.array([[0, -1], [1, 0]]) @ path_tangent

        dot_s = np.diag(veh_pos_from_path.T @ path_tangent)
        dot_e = np.diag(veh_pos_from_path.T @ path_normal)

        proj_s = dot_s / np.linalg.norm(path_tangent, axis=0, keepdims=True)
        proj_e = dot_e / np.linalg.norm(path_normal, axis=0, keepdims=True)

        result_s = matched_s + proj_s

        interp_dpsi = np.interp(result_s, self.s_m, self.psi_rad)
        result_dpsi = math.wrap_to_pi_float(psi_rad_1d - interp_dpsi)

        out_s_arr = result_s.squeeze()
        out_e_arr = proj_e.squeeze()
        out_dpsi_arr = result_dpsi.squeeze()

        try:
            out_s = out_s_arr.item()
            out_e = out_e_arr.item()
            out_dpsi = out_dpsi_arr.item()
        except ValueError:
            out_s = out_s_arr
            out_e = out_e_arr
            out_dpsi = out_dpsi_arr

        return out_s, out_e, out_dpsi

    def enpsi_to_sedpsi(
        self,
        east_m: np.ndarray,
        north_m: np.ndarray,
        psi_rad: np.ndarray,
        seed_s_m: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        window_size = _S_WINDOW
        if seed_s_m is None:
            seed_s_m = np.zeros(np.array(east_m).shape)
            window_size = self.length_m + 10

        arrays = _format_arrays([east_m, north_m, psi_rad, np.array(seed_s_m)])
        _check_array_sizes(arrays)
        east_m, north_m, psi_rad, seed_s_m = arrays

        query_array = np.column_stack((east_m, north_m))
        _, full_match_idx = self._tree.query(query_array, _NUM_KD_MATCHES)
        match_idx = self._find_closest_in_window(full_match_idx, seed_s_m, window_size)

        rel_east, rel_north = self._get_relative_pos(east_m, north_m, match_idx)
        psi_guess = self.psi_rad[match_idx]

        init_proj_s, _ = _project_veh_on_path(rel_east, rel_north, psi_guess)
        init_s = self.s_m[match_idx] + init_proj_s

        path_psi = np.interp(init_s, self.s_m, self.psi_rad)
        proj_s, proj_e = _project_veh_on_path(rel_east, rel_north, path_psi)

        result_s = self.s_m[match_idx] + proj_s
        result_dpsi = math.wrap_to_pi_float(psi_rad - path_psi)

        return _cast_array_list_to_floats(result_s, proj_e, result_dpsi)

    def seu_to_enu(
        self, s_m: np.ndarray, e_m: np.ndarray, dpsi_rad: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s_m_1d = np.array(s_m).squeeze()
        e_m_1d = np.array(e_m).squeeze()
        dpsi_rad_1d = np.array(dpsi_rad).squeeze()

        if (s_m_1d.ndim > 1) or (e_m_1d.ndim > 1) or (dpsi_rad_1d.ndim > 1):
            raise ValueError("All inputs must be row or column vectors")

        if not (s_m_1d.size == e_m_1d.size == dpsi_rad_1d.size):
            raise ValueError("All inputs must be the same size")

        left_neighbors = np.searchsorted(self.s_m, s_m_1d, side="left")
        lo_s_idx = np.clip(left_neighbors - 1, 0, np.Inf).astype(int).tolist()

        diff_s_m = s_m_1d - np.array(self.s_m)[lo_s_idx]

        nom_east_m = np.array(self.east_m)[lo_s_idx]
        nom_north_m = np.array(self.north_m)[lo_s_idx]
        nom_psi_rad = np.array(self.psi_rad)[lo_s_idx]

        out_east_m = (
            nom_east_m - e_m_1d * np.cos(nom_psi_rad) - diff_s_m * np.sin(nom_psi_rad)
        )
        out_north_m = (
            nom_north_m - e_m_1d * np.sin(nom_psi_rad) + diff_s_m * np.cos(nom_psi_rad)
        )

        interp_psi = np.interp(s_m, self.s_m, self.psi_rad).squeeze()
        out_psi_rad = math.wrap_to_pi_float(interp_psi + dpsi_rad_1d)

        return out_east_m, out_north_m, out_psi_rad

    def sedpsi_to_enpsi(
        self, s_m: np.ndarray, e_m: np.ndarray, dpsi_rad: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrays = _format_arrays([s_m, e_m, dpsi_rad])
        _check_array_sizes(arrays)
        s_m, e_m, dpsi_rad = arrays

        l_idx, r_idx = self._get_nearest_neighbors(s_m)
        l_diff_s = s_m - self.s_m[l_idx]
        r_diff_s = s_m - self.s_m[r_idx]

        idx = np.where(np.abs(l_diff_s) <= np.abs(r_diff_s), l_idx, r_idx)
        diff_s = s_m - self.s_m[idx]

        path_east = self.east_m[idx]
        path_north = self.north_m[idx]
        path_psi = np.interp(s_m, self.s_m, self.psi_rad)

        out_east = path_east - e_m * np.cos(path_psi) - diff_s * np.sin(path_psi)
        out_north = path_north - e_m * np.sin(path_psi) + diff_s * np.cos(path_psi)

        out_psi = math.wrap_to_pi_float(path_psi + dpsi_rad)

        return _cast_array_list_to_floats(out_east, out_north, out_psi)

    def double_field(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Double one field of the world object.
        """
        
        item = getattr(self, key)
        double_item = np.concatenate((item, item[1:]))

        return (self._double_s, double_item)

    def plot(self, ax: Axes, line_opts: dict[str, Any] = None):
        if line_opts is None:
            line_opts = {"linestyle": "--", "color": "black"}
        ax.plot(self.east_m, self.north_m, **line_opts)

    def asdict(self) -> dict[str, Any]:
        out_dict = dataclasses.asdict(self)

        for key in self._skip_fields:
            del out_dict[key]

        return out_dict

    def pickle(self, pickle_path: Path):
        with open(pickle_path, "wb") as file:
            pickle.dump(self.asdict(), file)

    def save_to_mat(self, mat_path: Path):
        io.savemat(file_name=mat_path, mdict=self.asdict())

    @classmethod
    @property
    def field_names(cls) -> tuple[str]:
        return tuple(
            item.name
            for item in dataclasses.fields(cls)
            if item.name not in cls._skip_fields
        )

    @classmethod
    def load_from_mat(cls, mat_path: Path) -> Self:
        world_dict = io.loadmat(mat_path, squeeze_me=True)

        del world_dict["__header__"]
        del world_dict["__globals__"]
        del world_dict["__version__"]

        return cls(**world_dict)

    @classmethod
    def load_from_pickle(cls, pickle_path: Path) -> Self:
        with open(pickle_path, "rb") as file:
            world_dict = pickle.load(file)

        return cls(**world_dict)

    @property
    def _double_s(self) -> np.ndarray:
        return np.concatenate((self.s_m, self.s_m[1:] + self.s_m[-1]))

    @property
    def _pos_array(self) -> np.ndarray:
        return np.column_stack((self.east_m, self.north_m))

    def _get_nearest_neighbors(self, s_m) -> tuple[int, int]:
        l_neighbor = np.searchsorted(self.s_m, s_m, side="left")

        l_idx = np.clip(l_neighbor - 1, 0, self.size - 1)
        r_idx = np.clip(l_neighbor, 0, self.size - 1)

        return l_idx, r_idx

    def __post_init__(self):
        _check_array_sizes([getattr(self, key) for key in self.field_names])
        object.__setattr__(self, "_tree", spatial.KDTree(self._pos_array))

    def __eq__(self, other: Self) -> bool:
        if not type(self) == type(other):
            return False

        for name in self.field_names:
            if not np.allclose(getattr(self, name), getattr(other, name)):
                return False

        return True

    def _get_relative_pos(
        self, east_m: np.ndarray, north_m: np.ndarray, match_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        path_east = self.east_m[match_idx]
        path_north = self.north_m[match_idx]

        del_east = east_m - path_east
        del_north = north_m - path_north

        return del_east, del_north

    def _find_closest_in_window(
        self, match_idx: np.ndarray, seed_s_m: np.ndarray, window_size: float
    ) -> np.ndarray:
        row, _ = match_idx.shape
        match_idx = match_idx[match_idx < self._tree.n].reshape((row, -1))
        dist_seed = np.abs(self.s_m[match_idx] - seed_s_m.reshape(-1, 1))
        window_bool = np.less(dist_seed, window_size)
        ok_idx = window_bool.argmax(axis=1)

        best_idx = []
        for row in range(seed_s_m.size):
            row_idx = match_idx[row, ok_idx[row]]
            best_idx.append(row_idx)

        return np.array(best_idx)

    @classmethod
    @property
    def _skip_fields(cls) -> set[str]:
        return {"_tree"}


def default_marker_opts() -> dict[str, Any]:
    return {
        "marker": "D",
        "c": "black",
    }


def default_label_opts() -> dict[str, Any]:
    return {
        "ha": "left",
        "va": "center",
        "clip_on": True,
        "bbox": {"boxstyle": "square,pad=0.3", "fc": "white", "ec": "black"},
    }


@dataclass
class LabelConfig:
    x_offset: float = 10.0
    y_offset: float = 0.0
    path_step: float = 100.0
    marker_opt: InitVar[dict[str, Any]] = None
    label_opt: InitVar[dict[str, Any]] = None

    m_opt: dict[str, Any] = field(init=False, default_factory=default_marker_opts)
    l_opt: dict[str, Any] = field(init=False, default_factory=default_label_opts)

    def __post_init__(self, marker_opt: dict[str, Any], label_opt: dict[str, Any]):
        if marker_opt is not None:
            self.m_opt.update(marker_opt)

        if label_opt is not None:
            self.l_opt.update(label_opt)


def _get_path_label_positions(world: SimpleWorld, step: float) -> list[float]:
    world_min_s = np.round(np.min(world.s_m))
    world_max_s = np.round(np.max(world.s_m))

    rounded_2nd_marker = np.ceil(world_min_s / step) * step
    s_label = np.arange(start=rounded_2nd_marker, stop=world_max_s, step=step)
    s_label = np.concatenate((np.array([world_min_s]), s_label)).tolist()

    return s_label


def plot_label(
    world: SimpleWorld, ax: Axes, s_pos: float, text: str, config: LabelConfig = None
):
    if config is None:
        config = LabelConfig()

    east_pos = np.interp(s_pos, world.s_m, world.east_m)
    north_pos = np.interp(s_pos, world.s_m, world.north_m)
    ax.scatter(east_pos, north_pos, **config.m_opt)
    ax.text(
        x=east_pos + config.x_offset,
        y=north_pos + config.y_offset,
        s=text,
        **config.l_opt,
    )


def plot_many_labels(
    world: SimpleWorld,
    ax: Axes,
    s_pos: list[float],
    labels: list[str],
    config: LabelConfig = None,
):

    if not len(s_pos) == len(labels):
        raise ValueError("s_pos and labels lists must be the same length")

    if config is None:
        config = LabelConfig()

    for pos, text in zip(s_pos, labels):
        plot_label(world=world, ax=ax, s_pos=pos, text=text, config=config)


def plot_path_distance_markers(
    ax: Axes, world: SimpleWorld, config: LabelConfig = None
):
    if config is None:
        config = LabelConfig()

    s_pos = _get_path_label_positions(world=world, step=config.path_step)
    labels = [f"s={s:,.0f}" for s in s_pos]
    plot_many_labels(world=world, ax=ax, s_pos=s_pos, labels=labels, config=config)


def rediscretize_s(world: SimpleWorld, ds_m: float) -> SimpleWorld:
    min_s, max_s = np.min(world.s_m), np.max(world.s_m)
    num = int(np.ceil((max_s - min_s) / ds_m)) + 1

    old_s = np.copy(world.s_m)
    new_s = np.linspace(min_s, max_s, num)

    wdict = world.asdict()
    for key, val in wdict.items():
        wdict[key] = np.interp(new_s, old_s, val)

    wdict["s_m"] = new_s
    return type(world)(**wdict)
