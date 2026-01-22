import math
import numpy as np


def lerp(a: float, b: float, d: float) -> float:
    return a + (b - a) * float(np.clip(d, 0.0, 1.0))


def meters_to_cells(val_m: float, h_scale: float, min_cells: int = 1) -> int:
    return max(min_cells, int(round(float(val_m) / float(h_scale))))


def _grid_coords(width: int, length: int, h_scale: float):
    cx = (width - 1) * 0.5
    cy = (length - 1) * 0.5
    x_coords = (np.arange(width) - cx) * h_scale
    y_coords = (np.arange(length) - cy) * h_scale
    return x_coords, y_coords


def stamp_box(hf: np.ndarray, cx: int, cy: int, sx: int, sy: int, height_cells: int):
    x1 = max(0, cx - sx // 2)
    x2 = min(hf.shape[0], cx + int(math.ceil(sx / 2)))
    y1 = max(0, cy - sy // 2)
    y2 = min(hf.shape[1], cy + int(math.ceil(sy / 2)))
    if x2 <= x1 or y2 <= y1:
        return
    hf[x1:x2, y1:y2] = np.maximum(hf[x1:x2, y1:y2], height_cells)


def stamp_cylinder(hf: np.ndarray, cx: int, cy: int, radius: int, height_cells: int):
    if radius <= 0:
        return
    x1 = max(0, cx - radius)
    x2 = min(hf.shape[0], cx + radius + 1)
    y1 = max(0, cy - radius)
    y2 = min(hf.shape[1], cy + radius + 1)
    if x2 <= x1 or y2 <= y1:
        return
    xs = np.arange(x1, x2) - cx
    ys = np.arange(y1, y2) - cy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    mask = (xx * xx + yy * yy) <= radius * radius
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


def _build_s1_corridor(hf, x_coords, y_coords, cfg, d, rng, meta):
    clearance = meta["clearance_m"]
    wall_h = meta["wall_height_m"]
    width_max = meta["corridor_width_max_m"]
    width_min = meta["corridor_width_min_m"]
    gate_w_max = meta["gate_width_max_m"]
    gate_w_min = meta["gate_width_min_m"]
    gate_len_max = meta["gate_length_max_m"]
    gate_len_min = meta["gate_length_min_m"]
    k_min = int(meta["gate_count_min"])
    k_max = int(meta["gate_count_max"])

    width_nom = lerp(width_max, width_min, d)
    width_nom = min(width_nom, meta["terrain_width_m"] - 2.0 * clearance)
    half_nom = 0.5 * width_nom

    gate_width = lerp(gate_w_max, gate_w_min, d)
    gate_width = max(gate_width, 2.2 * clearance)
    half_gate = 0.5 * gate_width
    gate_len = lerp(gate_len_max, gate_len_min, d)

    k = max(1, int(round(lerp(k_min, k_max, d))))
    y_min = float(y_coords[0] + gate_len)
    y_max = float(y_coords[-1] - gate_len)
    gates = []
    for _ in range(k):
        for _ in range(50):
            y0 = rng.uniform(y_min, y_max)
            if all(abs(y0 - gy) > gate_len for gy, _, _ in gates):
                gates.append((y0, gate_len, gate_width))
                break

    wall_cells = meters_to_cells(wall_h, meta["v_scale_m"])
    for yi, y in enumerate(y_coords):
        half_w = half_nom
        for y0, gl, gw in gates:
            if abs(y - y0) <= 0.5 * gl:
                half_w = min(half_w, 0.5 * gw)
        mask = np.abs(x_coords) > half_w
        hf[mask, yi] = np.maximum(hf[mask, yi], wall_cells)

    meta["gate_count"] = len(gates)
    meta["gate_width_m"] = gate_width
    meta["corridor_width_m"] = width_nom


def _build_s2_forest(hf, x_coords, y_coords, cfg, d, rng, meta):
    clearance = meta["clearance_m"]
    obs_h = meta["obs_height_m"]
    n_min = int(meta["forest_count_min"])
    n_max = int(meta["forest_count_max"])
    n = int(round(lerp(n_min, n_max, d)))
    size_min = meta["forest_size_min_m"]
    size_max = meta["forest_size_max_m"]
    block_ratio = meta["forest_block_ratio"]
    min_dist = meta["forest_min_dist_m"]

    bounds = (
        (float(x_coords[0] + clearance), float(x_coords[-1] - clearance)),
        (float(y_coords[0] + clearance), float(y_coords[-1] - clearance)),
    )
    points = _sample_points(rng, n, bounds, min_dist, max_tries=n * 50 + 500)
    h_cells = meters_to_cells(obs_h, meta["v_scale_m"])
    for x, y in points:
        size = lerp(size_min, size_max, rng.rand())
        cx = int(np.argmin(np.abs(x_coords - x)))
        cy = int(np.argmin(np.abs(y_coords - y)))
        if rng.rand() < block_ratio:
            sx = meters_to_cells(size, meta["h_scale_m"])
            stamp_box(hf, cx, cy, sx, sx, h_cells)
        else:
            r = meters_to_cells(0.5 * size, meta["h_scale_m"])
            stamp_cylinder(hf, cx, cy, r, h_cells)
    meta["obs_count"] = len(points)


def _build_s3_doorway(hf, x_coords, y_coords, cfg, d, rng, meta):
    clearance = meta["clearance_m"]
    wall_h = meta["wall_height_m"]
    door_w_max = meta["door_width_max_m"]
    door_w_min = meta["door_width_min_m"]
    wall_th = meta["door_wall_thickness_m"]
    walls_min = int(meta["door_wall_count_min"])
    walls_max = int(meta["door_wall_count_max"])
    walls_n = max(1, int(round(lerp(walls_min, walls_max, d))))
    door_w = lerp(door_w_max, door_w_min, d)
    door_w = max(door_w, 2.2 * clearance)
    wall_cells = meters_to_cells(wall_h, meta["v_scale_m"])
    th_cells = meters_to_cells(wall_th, meta["h_scale_m"])
    y_min = y_coords[0] + wall_th
    y_max = y_coords[-1] - wall_th

    walls = []
    for _ in range(walls_n):
        for _ in range(50):
            y0 = rng.uniform(y_min, y_max)
            if all(abs(y0 - wy) > 2.0 * wall_th for wy in walls):
                walls.append(y0)
                break

    for y0 in walls:
        y_idx = int(np.argmin(np.abs(y_coords - y0)))
        y1 = max(0, y_idx - th_cells // 2)
        y2 = min(hf.shape[1], y_idx + int(math.ceil(th_cells / 2)))
        door_center = 0.0
        half_door = 0.5 * door_w
        mask = np.abs(x_coords - door_center) > half_door
        hf[mask, y1:y2] = np.maximum(hf[mask, y1:y2], wall_cells)

    meta["door_count"] = len(walls)
    meta["door_width_m"] = door_w


def _build_s5_sparse_dense(hf, x_coords, y_coords, cfg, d, rng, meta):
    split = meta["s5_split_ratio"]
    y_split = int(round(hf.shape[1] * split))
    y_split = max(1, min(hf.shape[1] - 1, y_split))
    y_coords_a = y_coords[:y_split]
    y_coords_b = y_coords[y_split:]

    meta_a = dict(meta)
    meta_b = dict(meta)
    meta_a["forest_count_min"] = meta["s5_sparse_count_min"]
    meta_a["forest_count_max"] = meta["s5_sparse_count_max"]
    meta_b["forest_count_min"] = meta["s5_dense_count_min"]
    meta_b["forest_count_max"] = meta["s5_dense_count_max"]

    _build_s2_forest(hf[:, :y_split], x_coords, y_coords_a, cfg, d, rng, meta_a)
    _build_s2_forest(hf[:, y_split:], x_coords, y_coords_b, cfg, d, rng, meta_b)

    meta["obs_count"] = meta_a.get("obs_count", 0) + meta_b.get("obs_count", 0)


def _build_s6_cluster(hf, x_coords, y_coords, cfg, d, rng, meta):
    clearance = meta["clearance_m"]
    obs_h = meta["obs_height_m"]
    cluster_min = int(meta["cluster_count_min"])
    cluster_max = int(meta["cluster_count_max"])
    per_min = int(meta["cluster_size_min"])
    per_max = int(meta["cluster_size_max"])
    sigma_max = meta["cluster_sigma_max_m"]
    sigma_min = meta["cluster_sigma_min_m"]
    cluster_n = max(1, int(round(lerp(cluster_min, cluster_max, d))))
    sigma = lerp(sigma_max, sigma_min, d)
    h_cells = meters_to_cells(obs_h, meta["v_scale_m"])

    bounds = (
        (float(x_coords[0] + clearance), float(x_coords[-1] - clearance)),
        (float(y_coords[0] + clearance), float(y_coords[-1] - clearance)),
    )
    centers = _sample_points(rng, cluster_n, bounds, min_dist=2.0 * clearance, max_tries=500)
    total = 0
    for cx, cy in centers:
        count = rng.randint(per_min, per_max + 1)
        for _ in range(count):
            x = rng.normal(cx, sigma)
            y = rng.normal(cy, sigma)
            x = float(np.clip(x, bounds[0][0], bounds[0][1]))
            y = float(np.clip(y, bounds[1][0], bounds[1][1]))
            ix = int(np.argmin(np.abs(x_coords - x)))
            iy = int(np.argmin(np.abs(y_coords - y)))
            r = meters_to_cells(meta["forest_size_min_m"], meta["h_scale_m"])
            stamp_cylinder(hf, ix, iy, r, h_cells)
            total += 1
    meta["obs_count"] = total


def build_heightfield(scene_type: str, d: float, rng, cfg) -> dict:
    width = int(cfg.terrain_width / cfg.horizontal_scale)
    length = int(cfg.terrain_length / cfg.horizontal_scale)
    hf = np.zeros((width, length), dtype=np.int16)

    scene_cfg = getattr(cfg, "scene_cfg", {}) or {}
    clearance = float(scene_cfg.get("clearance_m", getattr(cfg, "scene_clearance", 0.27)))
    wall_k = float(scene_cfg.get("wall_height_k", 3.0))
    obs_k = float(scene_cfg.get("obs_height_k", 1.5))

    meta = {
        "scene_type": scene_type,
        "difficulty": float(d),
        "clearance_m": clearance,
        "terrain_width_m": float(cfg.terrain_width),
        "terrain_length_m": float(cfg.terrain_length),
        "h_scale_m": float(cfg.horizontal_scale),
        "v_scale_m": float(cfg.vertical_scale),
        "wall_height_m": wall_k * clearance,
        "obs_height_m": obs_k * clearance,
        "layout_id": int(scene_cfg.get("scene_seed", 0)),
    }

    x_coords, y_coords = _grid_coords(width, length, meta["h_scale_m"])

    if scene_type == "s1_corridor":
        meta.update(
            {
                "corridor_width_max_m": scene_cfg.get("corridor_width_max_k", 6.0) * clearance,
                "corridor_width_min_m": scene_cfg.get("corridor_width_min_k", 4.0) * clearance,
                "gate_width_max_m": scene_cfg.get("gate_width_max_k", 4.0) * clearance,
                "gate_width_min_m": scene_cfg.get("gate_width_min_k", 2.5) * clearance,
                "gate_length_max_m": scene_cfg.get("gate_length_max_k", 4.0) * clearance,
                "gate_length_min_m": scene_cfg.get("gate_length_min_k", 2.0) * clearance,
                "gate_count_min": scene_cfg.get("gate_count_min", 1),
                "gate_count_max": scene_cfg.get("gate_count_max", 3),
            }
        )
        _build_s1_corridor(hf, x_coords, y_coords, cfg, d, rng, meta)
    elif scene_type == "s2_forest":
        meta.update(
            {
                "forest_count_min": scene_cfg.get("forest_count_min", 12),
                "forest_count_max": scene_cfg.get("forest_count_max", 36),
                "forest_size_min_m": scene_cfg.get("forest_size_min_k", 1.2) * clearance,
                "forest_size_max_m": scene_cfg.get("forest_size_max_k", 2.0) * clearance,
                "forest_min_dist_m": scene_cfg.get("forest_min_dist_k", 2.0) * clearance,
                "forest_block_ratio": scene_cfg.get("forest_block_ratio", 0.5),
            }
        )
        _build_s2_forest(hf, x_coords, y_coords, cfg, d, rng, meta)
    elif scene_type == "s3_doorway":
        meta.update(
            {
                "door_width_max_m": scene_cfg.get("door_width_max_k", 5.0) * clearance,
                "door_width_min_m": scene_cfg.get("door_width_min_k", 3.0) * clearance,
                "door_wall_thickness_m": scene_cfg.get("door_wall_thickness_k", 1.0) * clearance,
                "door_wall_count_min": scene_cfg.get("door_wall_count_min", 1),
                "door_wall_count_max": scene_cfg.get("door_wall_count_max", 2),
            }
        )
        _build_s3_doorway(hf, x_coords, y_coords, cfg, d, rng, meta)
    elif scene_type == "s5_sparse_dense":
        meta.update(
            {
                "forest_size_min_m": scene_cfg.get("forest_size_min_k", 1.2) * clearance,
                "forest_size_max_m": scene_cfg.get("forest_size_max_k", 2.0) * clearance,
                "forest_min_dist_m": scene_cfg.get("forest_min_dist_k", 2.0) * clearance,
                "forest_block_ratio": scene_cfg.get("forest_block_ratio", 0.5),
                "s5_split_ratio": scene_cfg.get("s5_split_ratio", 0.5),
                "s5_sparse_count_min": scene_cfg.get("s5_sparse_count_min", 6),
                "s5_sparse_count_max": scene_cfg.get("s5_sparse_count_max", 18),
                "s5_dense_count_min": scene_cfg.get("s5_dense_count_min", 18),
                "s5_dense_count_max": scene_cfg.get("s5_dense_count_max", 42),
            }
        )
        _build_s5_sparse_dense(hf, x_coords, y_coords, cfg, d, rng, meta)
    elif scene_type == "s6_ood_structured":
        meta.update(
            {
                "forest_size_min_m": scene_cfg.get("forest_size_min_k", 1.2) * clearance,
                "cluster_count_min": scene_cfg.get("cluster_count_min", 2),
                "cluster_count_max": scene_cfg.get("cluster_count_max", 4),
                "cluster_size_min": scene_cfg.get("cluster_size_min", 6),
                "cluster_size_max": scene_cfg.get("cluster_size_max", 12),
                "cluster_sigma_min_m": scene_cfg.get("cluster_sigma_min_k", 1.0) * clearance,
                "cluster_sigma_max_m": scene_cfg.get("cluster_sigma_max_k", 2.5) * clearance,
            }
        )
        _build_s6_cluster(hf, x_coords, y_coords, cfg, d, rng, meta)

    return {"hf": hf, "meta": meta}
