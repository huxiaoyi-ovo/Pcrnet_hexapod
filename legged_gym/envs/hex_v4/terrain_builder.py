import math
import numpy as np


def lerp(a: float, b: float, d: float) -> float:
    return a + (b - a) * float(np.clip(d, 0.0, 1.0))


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def meters_to_grid(x_m: float, y_m: float, x_min: float, y_min: float, h_scale: float):
    ix = int(round((x_m - x_min) / h_scale))
    iy = int(round((y_m - y_min) / h_scale))
    return ix, iy


def _rect_indices(x0: float, x1: float, y0: float, y1: float, x_min: float, y_min: float,
                  h_scale: float, width: int, length: int):
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    ix0 = int(math.floor((x0 - x_min) / h_scale))
    ix1 = int(math.ceil((x1 - x_min) / h_scale))
    iy0 = int(math.floor((y0 - y_min) / h_scale))
    iy1 = int(math.ceil((y1 - y_min) / h_scale))
    ix0 = max(0, min(width, ix0))
    ix1 = max(0, min(width, ix1))
    iy0 = max(0, min(length, iy0))
    iy1 = max(0, min(length, iy1))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return ix0, ix1, iy0, iy1


def stamp_box(hf: np.ndarray, x0: float, x1: float, y0: float, y1: float, height_cells: int,
              x_min: float, y_min: float, h_scale: float):
    idx = _rect_indices(x0, x1, y0, y1, x_min, y_min, h_scale, hf.shape[0], hf.shape[1])
    if idx is None:
        return
    ix0, ix1, iy0, iy1 = idx
    hf[ix0:ix1, iy0:iy1] = np.maximum(hf[ix0:ix1, iy0:iy1], height_cells)


def clear_rect(hf: np.ndarray, x0: float, x1: float, y0: float, y1: float,
               x_min: float, y_min: float, h_scale: float):
    idx = _rect_indices(x0, x1, y0, y1, x_min, y_min, h_scale, hf.shape[0], hf.shape[1])
    if idx is None:
        return
    ix0, ix1, iy0, iy1 = idx
    hf[ix0:ix1, iy0:iy1] = 0


def stamp_cylinder(hf: np.ndarray, cx: float, cy: float, radius: float, height_cells: int,
                   x_min: float, y_min: float, h_scale: float):
    if radius <= 0.0:
        return
    ix, iy = meters_to_grid(cx, cy, x_min, y_min, h_scale)
    r_cells = max(1, int(round(radius / h_scale)))
    x1 = max(0, ix - r_cells)
    x2 = min(hf.shape[0], ix + r_cells + 1)
    y1 = max(0, iy - r_cells)
    y2 = min(hf.shape[1], iy + r_cells + 1)
    if x2 <= x1 or y2 <= y1:
        return
    xs = np.arange(x1, x2) - ix
    ys = np.arange(y1, y2) - iy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    mask = (xx * xx + yy * yy) <= r_cells * r_cells
    patch = hf[x1:x2, y1:y2]
    patch[mask] = np.maximum(patch[mask], height_cells)
    hf[x1:x2, y1:y2] = patch


def _sample_points(rng, n, bounds, min_dist, max_tries=2000):
    (x_min, x_max), (y_min, y_max) = bounds
    points = []
    tries = 0
    while len(points) < n and tries < max_tries:
        tries += 1
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        ok = True
        for px, py in points:
            if (x - px) ** 2 + (y - py) ** 2 < min_dist ** 2:
                ok = False
                break
        if ok:
            points.append((x, y))
    return points


def _resolve_dims(scene_w: float, scene_l: float, width_m: float, length_m: float):
    w = min(scene_w, width_m)
    l = min(scene_l, length_m)
    return w, l


def _build_s1_corridor(hf, cfg, d, rng, meta, x_min, y_min, h_scale):
    c = meta["clearance_m"]
    length_mul = cfg.get("length_mul", (18.0, 28.0))
    half_width_mul = cfg.get("half_width_mul", (6.0, 4.0))
    gate_half_width_mul = cfg.get("gate_half_width_mul", (4.0, 2.2))
    wall_thickness_mul = cfg.get("wall_thickness_mul", 2.0)
    gate_len_mul = cfg.get("gate_len_mul", (4.0, 2.5))
    gate_count = cfg.get("gate_count", (2, 5))
    gate_min_gap_mul = cfg.get("gate_min_gap_mul", 3.0)
    spawn_clear_mul = cfg.get("spawn_clear_mul", 3.0)
    goal_clear_mul = cfg.get("goal_clear_mul", 3.0)
    wall_height_mul = cfg.get("wall_height_mul", (2.5, 3.5))

    L = lerp(length_mul[0] * c, length_mul[1] * c, d)
    W0 = lerp(half_width_mul[0] * c, half_width_mul[1] * c, d)
    Wg = lerp(gate_half_width_mul[0] * c, gate_half_width_mul[1] * c, d)
    Tw = wall_thickness_mul * c
    Lg = lerp(gate_len_mul[0] * c, gate_len_mul[1] * c, d)
    k = max(1, int(round(lerp(gate_count[0], gate_count[1], d))))

    W0 = min(W0, meta["width_m"] * 0.5 - Tw - 0.1 * c)
    W0 = max(W0, 2.5 * c)
    Wg = min(Wg, W0)
    Wg = max(Wg, 2.2 * c)

    L = min(L, meta["length_m"])
    Lg = min(Lg, max(2.0 * c, 0.4 * L))

    wall_h = lerp(wall_height_mul[0] * c, wall_height_mul[1] * c, d)
    wall_cells = max(1, int(round(wall_h / meta["v_scale_m"])))

    # Base corridor walls
    stamp_box(hf, -(W0 + Tw), -W0, 0.0, L, wall_cells, x_min, y_min, h_scale)
    stamp_box(hf, W0, (W0 + Tw), 0.0, L, wall_cells, x_min, y_min, h_scale)

    gates = []
    y_low = 0.2 * L
    y_high = 0.8 * L
    if y_high - y_low < Lg:
        y_low = Lg
        y_high = max(y_low, L - Lg)
    gap_min = gate_min_gap_mul * c
    for _ in range(k):
        for _ in range(60):
            y0 = rng.uniform(y_low, y_high)
            if all(abs(y0 - gy) > (Lg + gap_min) for gy, _ in gates):
                gates.append((y0, Lg))
                break

    for y0, gl in gates:
        y_a = y0 - 0.5 * gl
        y_b = y0 + 0.5 * gl
        # Clear old walls
        clear_rect(hf, -(W0 + Tw), -W0, y_a, y_b, x_min, y_min, h_scale)
        clear_rect(hf, W0, (W0 + Tw), y_a, y_b, x_min, y_min, h_scale)
        # Draw inner walls
        stamp_box(hf, -(Wg + Tw), -Wg, y_a, y_b, wall_cells, x_min, y_min, h_scale)
        stamp_box(hf, Wg, (Wg + Tw), y_a, y_b, wall_cells, x_min, y_min, h_scale)

    spawn_clear = spawn_clear_mul * c
    goal_clear = goal_clear_mul * c
    clear_rect(hf, -W0, W0, 0.0, spawn_clear, x_min, y_min, h_scale)
    clear_rect(hf, -W0, W0, L - goal_clear, L, x_min, y_min, h_scale)

    meta.update({
        "L": L,
        "W0": W0,
        "Wg": Wg,
        "Tw": Tw,
        "gate_count": len(gates),
        "gates": [(gy - 0.5 * gl, gy + 0.5 * gl) for gy, gl in gates],
    })


def _build_s2_forest(hf, cfg, d, rng, meta, x_min, y_min, h_scale):
    c = meta["clearance_m"]
    length_mul = cfg.get("length_mul", (20.0, 30.0))
    width_mul = cfg.get("width_mul", (16.0, 20.0))
    count_range = cfg.get("count_range", (40, 160))
    min_dist_mul = cfg.get("min_dist_mul", (3.0, 2.0))
    block_size_mul = cfg.get("block_size_mul", (1.5, 2.5))
    pole_radius_mul = cfg.get("pole_radius_mul", (0.6, 1.0))
    block_ratio = cfg.get("block_ratio", 0.5)
    spawn_clear_mul = cfg.get("spawn_clear_mul", 3.0)
    obs_height_mul = cfg.get("obs_height_mul", (2.0, 3.0))

    L = lerp(length_mul[0] * c, length_mul[1] * c, d)
    W = lerp(width_mul[0] * c, width_mul[1] * c, d)
    L, W = _resolve_dims(W, L, meta["width_m"], meta["length_m"])

    n = int(round(lerp(count_range[0], count_range[1], d)))
    min_dist = lerp(min_dist_mul[0] * c, min_dist_mul[1] * c, d)
    spawn_clear = spawn_clear_mul * c
    h_obs = lerp(obs_height_mul[0] * c, obs_height_mul[1] * c, d)
    h_cells = max(1, int(round(h_obs / meta["v_scale_m"])))

    bounds = ((-0.5 * W, 0.5 * W), (spawn_clear, L))
    points = _sample_points(rng, n, bounds, min_dist, max_tries=n * 20 + 200)
    for x, y in points:
        if rng.rand() < block_ratio:
            size = lerp(block_size_mul[0] * c, block_size_mul[1] * c, rng.rand())
            stamp_box(hf, x - 0.5 * size, x + 0.5 * size, y - 0.5 * size, y + 0.5 * size,
                      h_cells, x_min, y_min, h_scale)
        else:
            r = lerp(pole_radius_mul[0] * c, pole_radius_mul[1] * c, rng.rand())
            stamp_cylinder(hf, x, y, r, h_cells, x_min, y_min, h_scale)

    meta.update({
        "L": L,
        "W": W,
        "N": n,
        "placed": len(points),
        "min_dist": min_dist,
        "spawn_clear": spawn_clear,
    })


def _build_s3_doorway(hf, cfg, d, rng, meta, x_min, y_min, h_scale):
    c = meta["clearance_m"]
    length_mul = cfg.get("length_mul", (24.0, 32.0))
    width_mul = cfg.get("width_mul", (18.0, 22.0))
    wall_thickness_mul = cfg.get("wall_thickness_mul", 2.0)
    door_width_mul = cfg.get("door_width_mul", (4.0, 2.2))
    wall_count = cfg.get("wall_count", (2, 5))
    outer_walls = cfg.get("outer_walls", True)
    door_zigzag = cfg.get("door_zigzag", True)
    wall_height_mul = cfg.get("wall_height_mul", (2.5, 3.5))

    L = lerp(length_mul[0] * c, length_mul[1] * c, d)
    W = lerp(width_mul[0] * c, width_mul[1] * c, d)
    L, W = _resolve_dims(W, L, meta["width_m"], meta["length_m"])

    Tw = wall_thickness_mul * c
    Wd = lerp(door_width_mul[0] * c, door_width_mul[1] * c, d)
    Wd = max(Wd, 2.2 * c)
    m = max(1, int(round(lerp(wall_count[0], wall_count[1], d))))
    wall_h = lerp(wall_height_mul[0] * c, wall_height_mul[1] * c, d)
    wall_cells = max(1, int(round(wall_h / meta["v_scale_m"])))

    if outer_walls:
        stamp_box(hf, -0.5 * W, -0.5 * W + Tw, 0.0, L, wall_cells, x_min, y_min, h_scale)
        stamp_box(hf, 0.5 * W - Tw, 0.5 * W, 0.0, L, wall_cells, x_min, y_min, h_scale)

    y_positions = np.linspace(0.2 * L, 0.8 * L, m)
    doors = []
    for i, y0 in enumerate(y_positions):
        y0 = float(y0 + rng.uniform(-0.03 * L, 0.03 * L))
        stamp_box(hf, -0.5 * W, 0.5 * W, y0 - 0.5 * Tw, y0 + 0.5 * Tw,
                  wall_cells, x_min, y_min, h_scale)
        if door_zigzag:
            x_center = (-0.25 * W) if (i % 2 == 0) else (0.25 * W)
        else:
            x_center = 0.0
        x_center += rng.uniform(-0.05 * W, 0.05 * W)
        clear_rect(hf, x_center - 0.5 * Wd, x_center + 0.5 * Wd,
                   y0 - 0.5 * Tw, y0 + 0.5 * Tw, x_min, y_min, h_scale)
        doors.append((y0, x_center))

    meta.update({
        "L": L,
        "W": W,
        "m": m,
        "Wd": Wd,
        "doors": doors,
    })


def _build_s5_sparse_dense(hf, cfg, d, rng, meta, x_min, y_min, h_scale):
    c = meta["clearance_m"]
    length_mul = cfg.get("length_mul", (30.0, 30.0))
    width_mul = cfg.get("width_mul", (20.0, 20.0))
    split_range = cfg.get("split_range", (0.35, 0.65))
    sparse_count = cfg.get("sparse_count", (20, 60))
    dense_count = cfg.get("dense_count", (80, 180))
    min_sparse = cfg.get("min_dist_sparse_mul", (3.5, 2.8))
    min_dense = cfg.get("min_dist_dense_mul", (2.5, 1.8))
    block_size_mul = cfg.get("block_size_mul", (1.5, 2.5))
    pole_radius_mul = cfg.get("pole_radius_mul", (0.6, 1.0))
    block_ratio = cfg.get("block_ratio", 0.5)
    spawn_clear_mul = cfg.get("spawn_clear_mul", 3.0)
    obs_height_mul = cfg.get("obs_height_mul", (2.0, 3.0))

    L = lerp(length_mul[0] * c, length_mul[1] * c, d)
    W = lerp(width_mul[0] * c, width_mul[1] * c, d)
    L, W = _resolve_dims(W, L, meta["width_m"], meta["length_m"])

    y_split = rng.uniform(split_range[0] * L, split_range[1] * L)
    N1 = int(round(lerp(sparse_count[0], sparse_count[1], d)))
    N2 = int(round(lerp(dense_count[0], dense_count[1], d)))
    min1 = lerp(min_sparse[0] * c, min_sparse[1] * c, d)
    min2 = lerp(min_dense[0] * c, min_dense[1] * c, d)
    spawn_clear = spawn_clear_mul * c
    h_obs = lerp(obs_height_mul[0] * c, obs_height_mul[1] * c, d)
    h_cells = max(1, int(round(h_obs / meta["v_scale_m"])))

    bounds_a = ((-0.5 * W, 0.5 * W), (spawn_clear, y_split))
    bounds_b = ((-0.5 * W, 0.5 * W), (y_split, L))
    pts_a = _sample_points(rng, N1, bounds_a, min1, max_tries=N1 * 20 + 200)
    pts_b = _sample_points(rng, N2, bounds_b, min2, max_tries=N2 * 20 + 200)

    def stamp_points(points):
        for x, y in points:
            if rng.rand() < block_ratio:
                size = lerp(block_size_mul[0] * c, block_size_mul[1] * c, rng.rand())
                stamp_box(hf, x - 0.5 * size, x + 0.5 * size, y - 0.5 * size, y + 0.5 * size,
                          h_cells, x_min, y_min, h_scale)
            else:
                r = lerp(pole_radius_mul[0] * c, pole_radius_mul[1] * c, rng.rand())
                stamp_cylinder(hf, x, y, r, h_cells, x_min, y_min, h_scale)

    stamp_points(pts_a)
    stamp_points(pts_b)

    meta.update({
        "L": L,
        "W": W,
        "y_split": y_split,
        "N1": N1,
        "N2": N2,
        "placed1": len(pts_a),
        "placed2": len(pts_b),
    })


def _build_s6_cluster(hf, cfg, d, rng, meta, x_min, y_min, h_scale):
    c = meta["clearance_m"]
    length_mul = cfg.get("length_mul", (20.0, 30.0))
    width_mul = cfg.get("width_mul", (16.0, 20.0))
    cluster_count = cfg.get("cluster_count", (3, 6))
    cluster_radius_mul = cfg.get("cluster_radius_mul", (4.0, 6.0))
    cluster_size = cfg.get("cluster_size", (10, 30))
    cluster_sigma_mul = cfg.get("cluster_sigma_mul", (1.5, 2.5))
    block_size_mul = cfg.get("block_size_mul", (1.5, 2.5))
    pole_radius_mul = cfg.get("pole_radius_mul", (0.6, 1.0))
    block_ratio = cfg.get("block_ratio", 0.5)
    obs_height_mul = cfg.get("obs_height_mul", (2.0, 3.0))

    L = lerp(length_mul[0] * c, length_mul[1] * c, d)
    W = lerp(width_mul[0] * c, width_mul[1] * c, d)
    L, W = _resolve_dims(W, L, meta["width_m"], meta["length_m"])

    K = max(1, int(round(lerp(cluster_count[0], cluster_count[1], d))))
    R = lerp(cluster_radius_mul[0] * c, cluster_radius_mul[1] * c, d)
    sigma = lerp(cluster_sigma_mul[0] * c, cluster_sigma_mul[1] * c, d)
    h_obs = lerp(obs_height_mul[0] * c, obs_height_mul[1] * c, d)
    h_cells = max(1, int(round(h_obs / meta["v_scale_m"])))

    bounds = ((-0.5 * W, 0.5 * W), (0.0, L))
    centers = _sample_points(rng, K, bounds, min_dist=2.0 * R, max_tries=500)
    total = 0
    for cx, cy in centers:
        count = rng.randint(cluster_size[0], cluster_size[1] + 1)
        for _ in range(count):
            x = float(np.clip(rng.normal(cx, sigma), -0.5 * W, 0.5 * W))
            y = float(np.clip(rng.normal(cy, sigma), 0.0, L))
            if rng.rand() < block_ratio:
                size = lerp(block_size_mul[0] * c, block_size_mul[1] * c, rng.rand())
                stamp_box(hf, x - 0.5 * size, x + 0.5 * size, y - 0.5 * size, y + 0.5 * size,
                          h_cells, x_min, y_min, h_scale)
            else:
                r = lerp(pole_radius_mul[0] * c, pole_radius_mul[1] * c, rng.rand())
                stamp_cylinder(hf, x, y, r, h_cells, x_min, y_min, h_scale)
            total += 1

    meta.update({
        "L": L,
        "W": W,
        "K": len(centers),
        "R": R,
        "placed": total,
    })


def _build_s4_crossing_static(hf, cfg, d, rng, meta, x_min, y_min, h_scale):
    c = meta["clearance_m"]
    length_mul = cfg.get("length_mul", (20.0, 30.0))
    width_mul = cfg.get("width_mul", (16.0, 20.0))
    wall_band_mul = cfg.get("wall_band_mul", 2.0)
    gap_count = int(round(cfg.get("gap_count", 2)))
    gap_width_mul = cfg.get("gap_width_mul", 2.5)
    wall_height_mul = cfg.get("wall_height_mul", (2.0, 3.0))

    L = lerp(length_mul[0] * c, length_mul[1] * c, d)
    W = lerp(width_mul[0] * c, width_mul[1] * c, d)
    L, W = _resolve_dims(W, L, meta["width_m"], meta["length_m"])

    band_h = lerp(wall_height_mul[0] * c, wall_height_mul[1] * c, d)
    h_cells = max(1, int(round(band_h / meta["v_scale_m"])))
    band_th = wall_band_mul * c
    y0 = 0.6 * L
    stamp_box(hf, -0.5 * W, 0.5 * W, y0 - 0.5 * band_th, y0 + 0.5 * band_th,
              h_cells, x_min, y_min, h_scale)

    if gap_count > 0:
        gap_w = gap_width_mul * c
        xs = np.linspace(-0.3 * W, 0.3 * W, gap_count)
        for gx in xs:
            clear_rect(hf, gx - 0.5 * gap_w, gx + 0.5 * gap_w,
                       y0 - 0.5 * band_th, y0 + 0.5 * band_th, x_min, y_min, h_scale)

    meta.update({
        "L": L,
        "W": W,
        "band_y": y0,
        "gap_count": gap_count,
    })


def build_heightfield(scene_type: str, d: float, rng, cfg, horizontal_scale: float, vertical_scale: float):
    h_scale = float(horizontal_scale)
    v_scale = float(vertical_scale)
    width_m = float(cfg.terrain_width)
    length_m = float(cfg.terrain_length)
    width = max(1, int(round(width_m / h_scale)))
    length = max(1, int(round(length_m / h_scale)))
    hf = np.zeros((width, length), dtype=np.int16)

    scene_cfg = getattr(cfg, "scene_cfg", {}) or {}
    clearance = float(scene_cfg.get("clearance", getattr(cfg, "scene_clearance", 0.27)))

    meta = {
        "scene_type": scene_type,
        "difficulty": float(d),
        "clearance_m": clearance,
        "width_m": width_m,
        "length_m": length_m,
        "h_scale_m": h_scale,
        "v_scale_m": v_scale,
    }

    x_min = -0.5 * width_m
    y_min = 0.0

    if scene_type == "s1_corridor":
        _build_s1_corridor(hf, scene_cfg, d, rng, meta, x_min, y_min, h_scale)
    elif scene_type == "s2_forest":
        _build_s2_forest(hf, scene_cfg, d, rng, meta, x_min, y_min, h_scale)
    elif scene_type == "s3_doorway":
        _build_s3_doorway(hf, scene_cfg, d, rng, meta, x_min, y_min, h_scale)
    elif scene_type in ("s4_crossing", "s4_crossing_static"):
        _build_s4_crossing_static(hf, scene_cfg, d, rng, meta, x_min, y_min, h_scale)
    elif scene_type == "s5_sparse_dense":
        _build_s5_sparse_dense(hf, scene_cfg, d, rng, meta, x_min, y_min, h_scale)
    elif scene_type in ("s6_ood_cluster", "s6_ood_structured"):
        _build_s6_cluster(hf, scene_cfg, d, rng, meta, x_min, y_min, h_scale)

    return hf.astype(np.int16), meta
