"""
EDGS-like Dense COLMAP Initializer (concise & faithful)

Usage:
  python edgs_dense_init.py \
    --scene_root /path/to/scene \
    --images_subdir images_4 \
    --num_refs 150 --nns_per_ref 3 --matches_per_ref 20000 \
    --roma_model outdoor --certainty_thresh 0.2 --reproj_thresh 3.0 \
    --sampson_thresh 6.0 --min_parallax_deg 0.1 \
    --out_name points3D_dense.bin --viz

Main ideas transplanted from EDGS:
- Multi-NN per-pixel aggregation (argmax of certainty across neighbors)
- Certainty capping at sample_thresh, then multinomial sampling
- Geometry-aware filtering (Sampson, reprojection, cheirality, parallax)
- Colors from the reference view
"""

import os, argparse, struct, time
from typing import Dict, Tuple, List, Optional
import math
from collections import deque

import numpy as np
from PIL import Image
from tqdm import tqdm

# --------- IO / deps ----------
import pycolmap

# SciPy for k-means pose clustering
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist

# Optional viz
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False

# RoMa (romatch)
_ROMA_OK = False
try:
    import torch
    import torch.nn.functional as F
    from romatch import roma_outdoor, roma_indoor
    _ROMA_OK = True
except Exception:
    _ROMA_OK = False


# ==========================
# Small utilities
# ==========================

def log(s): print(f"[edgs-init] {s}")

def image_dir(scene_root: str, preferred: str) -> str:
    cand = os.path.join(scene_root, preferred)
    if os.path.isdir(cand): return cand
    for alt in ["images_4", "images_2", "images_8", "images"]:
        p = os.path.join(scene_root, alt)
        if os.path.isdir(p): return p
    raise FileNotFoundError("Could not locate an images directory under scene_root.")

def to_uint8_rgb(arr_float01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(arr_float01 * 255.0), 0, 255).astype(np.uint8)

def find_image(root: str, name: str) -> str:
    p = os.path.join(root, name)
    if os.path.isfile(p): return p
    q = os.path.join(root, os.path.basename(name))
    if os.path.isfile(q): return q
    raise FileNotFoundError(f"Image '{name}' not found under {root}")


# ==========================
# COLMAP helpers
# ==========================

def load_reconstruction(sparse_dir: str):
    rec = pycolmap.Reconstruction(sparse_dir)
    return rec, rec.cameras, rec.images

def K_from_camera(cam: pycolmap.Camera) -> np.ndarray:
    # EDGS expects correct mapping for SIMPLE_RADIAL/RADIAL/etc.
    K = np.eye(3, dtype=np.float64)
    model = str(cam.model.name).upper()
    p = np.asarray(cam.params, dtype=np.float64)
    w, h = cam.width, cam.height
    if "PINHOLE" in model and "SIMPLE" not in model:  # PINHOLE
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif "SIMPLE_PINHOLE" in model:
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif "SIMPLE_RADIAL" in model or model == "RADIAL":
        fx = fy = p[0]; cx, cy = p[1], p[2]  # ignore distortion for triangulation
    elif "OPENCV" in model:  # OPENCV*, FISHEYE, etc. -> first 4 are fx,fy,cx,cy
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        # Safe fallback: [f, cx, cy]
        fx = fy = p[0]
        cx = p[1] if len(p) > 1 else w/2
        cy = p[2] if len(p) > 2 else h/2
    K[0,0], K[1,1], K[0,2], K[1,2] = fx, fy, cx, cy
    return K

def pose_world2cam(im: pycolmap.Image) -> Tuple[np.ndarray, np.ndarray]:
    # world->cam rotation & translation
    if hasattr(im, "cam_from_world"):
        cfw = im.cam_from_world
        cfw = cfw() if callable(cfw) else cfw
        R = np.asarray(cfw.rotation.matrix(), dtype=np.float64)
        t = np.asarray(cfw.translation, dtype=np.float64).reshape(3,1)
    else:
        R = im.qvec.to_rotation_matrix()
        t = np.asarray(im.tvec, dtype=np.float64).reshape(3,1)
    return R, t

def P_from_KRt(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.concatenate([R, t], axis=1)  # 3x4

def cam_center_world(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (-R.T @ t).reshape(3)

def skew(v: np.ndarray) -> np.ndarray:
    vx, vy, vz = v.flatten()
    return np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]], dtype=np.float64)


# ==========================
# Neighbor selection
# ==========================

def select_cameras_by_visibility(rec: pycolmap.Reconstruction, K: int) -> List[int]:
    """
    Selects K reference cameras by greedily picking the one that sees the most
    not-yet-covered 3D points from the sparse reconstruction. This generally
    gives better spatial coverage than clustering poses alone.

    Args:
        rec: A pycolmap.Reconstruction object.
        K: The number of reference cameras to select.

    Returns:
        A sorted list of image_id's for the selected reference views.
    """
    if not rec.points3D:
        raise ValueError("Visibility-based selection requires a sparse point cloud.")

    # Map point3D_id -> list of observing image_ids
    pt_to_imgs = {pid: [el.image_id for el in p.track.elements] for pid, p in rec.points3D.items()}
    # Map image_id -> list of observed point3D_ids
    img_to_pts = {
        img.image_id: [p.point3D_id for p in img.points2D if p.has_point3D()]
        for img in rec.images.values()
    }
    img_to_pts = {iid: [p for p in pids if p != -1] for iid, pids in img_to_pts.items()}

    K = min(K, len(img_to_pts))
    
    # Greedily select cameras
    selected_cams = []
    covered_pts = set()

    # Score is the number of new points a camera would cover
    scores = {iid: len(pids) for iid, pids in img_to_pts.items()}

    for _ in range(K):
        if not scores: break
        # Pick camera that sees the most uncovered points
        best_cam = max(scores, key=scores.get)
        
        # Add to selected set and update covered points
        selected_cams.append(best_cam)
        newly_covered = set(img_to_pts[best_cam]) - covered_pts
        covered_pts.update(newly_covered)
        
        # Update scores for remaining cameras
        del scores[best_cam]
        for cam_id, cam_pts in img_to_pts.items():
            if cam_id in scores:
                # Penalize already covered points by removing them from consideration
                scores[cam_id] = len(set(cam_pts) - covered_pts)

    return sorted(selected_cams)

def select_cameras_kmeans(flat_poses: np.ndarray, K: int) -> List[int]:
    """
    K-centers (farthest-point sampling) over pose vectors.
    Drop-in replacement for the previous k-means medoid picker.

    Args:
        flat_poses: (N, D) array of pose embeddings (e.g., flattened 4x4s).
        K: number of references to select.

    Returns:
        Sorted list of N-local indices of selected reference views.
    """
    X = np.asarray(flat_poses, dtype=np.float32)
    N = X.shape[0]
    K = max(1, min(int(K), N))

    # Normalize features so dimensions contribute comparably.
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - mu) / sigma

    # Deterministic start: pick the pose farthest from the (normalized) centroid.
    # (Since Xn is zero-mean-ish, this is just the max norm row.)
    first = int(np.argmax(np.einsum("nd,nd->n", Xn, Xn)))
    centers = [first]

    # Track distance to nearest chosen center; update greedily.
    # dist[i] = min_j ||Xn[i] - Xn[centers[j]]||_2
    dist = np.linalg.norm(Xn - Xn[first], axis=1)
    dist[first] = -np.inf  # ensure we don't pick it again

    for _ in range(1, K):
        nxt = int(np.argmax(dist))
        centers.append(nxt)
        d = np.linalg.norm(Xn - Xn[nxt], axis=1)
        dist = np.minimum(dist, d)
        dist[nxt] = -np.inf

    return sorted(centers)

def nearest_neighbors(flat_poses: np.ndarray, k: int) -> np.ndarray:
    import torch as _torch
    M = _torch.from_numpy(flat_poses.astype(np.float32))
    with _torch.no_grad():
        D = _torch.cdist(M, M, p=2)
        D.fill_diagonal_(float("inf"))
        _, idx = _torch.topk(D, k, largest=False, dim=1)
    return idx.numpy()  # [N,k]


# ==========================
# RoMa wrapper (returns grid tensors)
# ==========================

class RomaMatcher:
    def __init__(self, device="cuda", mode="outdoor"):
        assert _ROMA_OK, "romatch not available"
        self.device = device
        self.model = roma_outdoor(device=device) if mode == "outdoor" else roma_indoor(device=device)
        self.model.eval()
        self.model.upsample_preds = False
        self.model.symmetric = False
        self.sample_thresh = getattr(self.model, "sample_thresh", 0.9)  # cap
        # warm-up
        W, H = self.model.w_resized, self.model.h_resized
        dummy = Image.new("RGB", (W, H), (0,0,0))
        _ = self.match_grids(dummy, dummy)

    @torch.inference_mode()
    def match_grids(self, imA: Image.Image, imB: Image.Image):
        # Return warp grid [H,W,4] and certainty grid [H,W] in [-1..1] normalized coordinates
        flow_or_warp, cert = self.model.match(imA, imB, device=self.device)
        # to tensors on device
        def TT(x):
            if isinstance(x, np.ndarray): x = torch.from_numpy(x)
            return x.to(self.device)
        FW, C = TT(flow_or_warp), TT(cert)

        # normalize shapes to [H,W,C] / [H,W]
        def to_hwC(t, cset=(2,4)):
            if t.ndim == 4 and t.shape[0] == 1 and t.shape[1] in cset:
                t = t[0].permute(1,2,0)
            elif t.ndim == 3 and t.shape[0] in cset:
                t = t.permute(1,2,0)
            elif t.ndim == 3 and t.shape[2] in cset:
                pass
            elif t.ndim == 2 and t.shape[1] in cset:
                ws, hs = self.model.w_resized, self.model.h_resized
                t = t.reshape(hs, ws, t.shape[1])
            else:
                raise RuntimeError(f"Unexpected flow/warp shape {tuple(t.shape)}")
            return t

        def to_hw1(c):
            if c.ndim == 4 and c.shape[0] == 1 and c.shape[1] == 1:
                c = c[0,0]
            elif c.ndim == 2:
                pass
            elif c.ndim == 1:
                ws, hs = self.model.w_resized, self.model.h_resized
                c = c.reshape(hs, ws)
            else:
                raise RuntimeError(f"Unexpected certainty shape {tuple(c.shape)}")
            return c

        FW = to_hwC(FW)          # [H,W,2] or [H,W,4]
        C  = to_hw1(C).sigmoid() # [H,W] in [0,1]

        H, W = C.shape
        yy = torch.linspace(-1 + 1/H, 1 - 1/H, H, device=self.device)
        xx = torch.linspace(-1 + 1/W, 1 - 1/W, W, device=self.device)
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        base = torch.stack([xx, yy], dim=-1)  # [H,W,2]

        if FW.shape[-1] == 4:      # already absolute (xA,yA,xB,yB)
            warp = FW
        else:                       # delta flow: (dx,dy)
            warp = torch.cat([base, base + FW], dim=-1)

        return warp.contiguous(), C.contiguous()  # [H,W,4], [H,W]


# ==========================
# Geometry & filters
# ==========================

def get_camera_axes(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the camera's principal axes in world coordinates."""
    x_axis = R.T[:, 0]  # Right vector
    y_axis = R.T[:, 1]  # Up vector
    z_axis = R.T[:, 2]  # Forward (viewing) vector
    return x_axis, y_axis, z_axis

def parallax_angle_for_cameras(R1: np.ndarray, R2: np.ndarray) -> float:
    """Computes the angle between the viewing axes of two cameras."""
    _, _, z1 = get_camera_axes(R1)
    _, _, z2 = get_camera_axes(R2)
    dot_product = np.clip(np.dot(z1, z2), -1.0, 1.0)
    return np.arccos(dot_product) # radians

@torch.inference_mode()
def match_and_filter_pair(
    matcher: "RomaMatcher",
    imA: Image.Image,
    imB: Image.Image,
    parallax_rad: float,
    cycle_thresh_px: float = 1.5,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Performs robust, cycle-consistent matching between two images.

    Args:
        matcher: The RomaMatcher instance.
        imA: Reference image.
        imB: Neighbor image.
        parallax_rad: Parallax angle in radians, used for weighting.
        cycle_thresh_px: Maximum allowed cycle consistency error in pixels.

    Returns:
        A tuple of (warp, weight), where warp is a [H, W, 4] tensor of
        valid matches and weight is a [H, W] tensor of their scores.
        Returns (None, None) if matching fails.
    """
    device = matcher.device
    w_match, h_match = matcher.model.w_resized, matcher.model.h_resized

    # Forward and backward matching
    try:
        warp_AB, cert_AB = matcher.match_grids(imA, imB)
        warp_BA, _ = matcher.match_grids(imB, imA)
    except Exception:
        return None, None

    # --- Cycle consistency check ---
    # `warp_AB` maps grid points in A to locations in B.
    # We need to sample `warp_BA` at these locations in B to get back to A.
    coords_B_normalized = warp_AB[..., 2:].clone()  # [H, W, 2] in [-1, 1]

    # `grid_sample` requires input grid of shape [N, H_in, W_in, 2]
    # and samples from a tensor of shape [N, C, H_out, W_out].
    # Here, we sample the "return map" of warp_BA using the destination
    # coordinates from warp_AB.
    warp_BA_return_map = warp_BA[..., 2:].permute(2, 0, 1).unsqueeze(0) # [1, 2, H, W]
    coords_B_grid = coords_B_normalized.unsqueeze(0) # [1, H, W, 2]

    # Sample the backward warp field at the forward-warped locations
    coords_A_reprojected = F.grid_sample(
        warp_BA_return_map,
        coords_B_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    ) # Result is [1, 2, H, W]
    coords_A_reprojected = coords_A_reprojected.squeeze(0).permute(1, 2, 0) # [H, W, 2]

    # Original coordinates in A are just the base grid
    yy = torch.linspace(-1 + 1/h_match, 1 - 1/h_match, h_match, device=device)
    xx = torch.linspace(-1 + 1/w_match, 1 - 1/w_match, w_match, device=device)
    coords_A_original = torch.stack(torch.meshgrid(yy, xx, indexing="ij"), dim=-1).flip(-1) # to XY

    # Calculate cycle error in pixels
    # Difference is in [-2, 2] range, scale to pixels
    cycle_dist_normalized = torch.linalg.norm(coords_A_original - coords_A_reprojected, dim=-1)
    scale_factor = torch.tensor([w_match, h_match], device=device).max() / 2.0
    cycle_dist_px = cycle_dist_normalized * scale_factor

    # --- Create mask and weights ---
    valid_mask = cycle_dist_px < cycle_thresh_px

    # Weight: combine RoMa certainty and parallax
    # Use sin(parallax) to favor larger baselines, discouraging collinear views.
    weight = cert_AB * torch.sin(torch.tensor(parallax_rad, device=device))
    weight[~valid_mask] = 0.0

    return warp_AB, weight


def dlt_triangulate_batch(P1: np.ndarray, P2: np.ndarray, uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    N = uv1.shape[0]
    X = np.zeros((N,4), dtype=np.float64)
    p10,p11,p12 = P1[0],P1[1],P1[2]
    p20,p21,p22 = P2[0],P2[1],P2[2]
    for i in range(N):
        u1,v1 = uv1[i]; u2,v2 = uv2[i]
        A = np.stack([
            u1*p12 - p10,
            v1*p12 - p11,
            u2*p22 - p20,
            v2*p22 - p21
        ], axis=0)
        _,_,Vt = np.linalg.svd(A)
        Xh = Vt[-1]
        if abs(Xh[3]) < 1e-12: Xh[3] = 1e-12
        X[i] = Xh / Xh[3]
    return X

def reprojection_errors(P: np.ndarray, X: np.ndarray, uv: np.ndarray) -> np.ndarray:
    Xh = X.T
    proj = (P @ Xh)
    proj = proj / np.maximum(1e-12, proj[2:3,:])
    pred = proj[:2,:].T
    return np.linalg.norm(pred - uv, axis=1)

def cheirality_mask(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xh = X.T
    z = (P @ Xh)[2,:]
    return (z > 0.0)

def parallax_mask(C1: np.ndarray, C2: np.ndarray, X: np.ndarray, min_deg=0.1) -> np.ndarray:
    v1 = X[:,:3] - C1.reshape(1,3)
    v2 = X[:,:3] - C2.reshape(1,3)
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True) + 1e-12
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True) + 1e-12
    ang = np.degrees(np.arccos(np.clip(np.sum(v1*v2,axis=1), -1.0, 1.0)))
    return ang >= float(min_deg)

def fundamental_from_world2cam(K1,R1,t1,K2,R2,t2) -> np.ndarray:
    # relative pose from cam1 to cam2 given world->cam
    R = R2 @ R1.T
    t = (t2 - R @ t1).reshape(3)
    E = skew(t) @ R
    K1i = np.linalg.inv(K1)
    K2i = np.linalg.inv(K2)
    F = K2i.T @ E @ K1i
    return F

def sampson_error(F: np.ndarray, uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    # uv are Nx2 in pixel coords; convert to homogeneous
    N = uv1.shape[0]
    x1 = np.concatenate([uv1, np.ones((N,1))], axis=1)  # Nx3
    x2 = np.concatenate([uv2, np.ones((N,1))], axis=1)
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    x2Fx1 = np.sum(x2 * Fx1, axis=1)
    den = Fx1[:,0]**2 + Fx1[:,1]**2 + Ftx2[:,0]**2 + Ftx2[:,1]**2 + 1e-12
    return (x2Fx1**2) / den


# ==========================
# Voxel downsample (optional)
# ==========================

def voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, voxel: float):
    if voxel <= 0: return xyz, rgb
    vids = np.floor(xyz / voxel).astype(np.int64)
    keys = vids[:,0]*73856093 ^ vids[:,1]*19349663 ^ vids[:,2]*83492791
    order = np.argsort(keys)
    keys_sorted = keys[order]
    xyz_sorted = xyz[order]; rgb_sorted = rgb[order]
    edge = np.nonzero(np.diff(keys_sorted))[0] + 1
    boundaries = np.concatenate([[0], edge, [len(keys_sorted)]])
    centers, colors = [], []
    for s,e in zip(boundaries[:-1], boundaries[1:]):
        centers.append(xyz_sorted[s:e].mean(axis=0))
        colors.append(rgb_sorted[s:e].mean(axis=0))
    return np.asarray(centers), np.asarray(colors)


# ==========================
# Writer: points3D.bin
# ==========================

def write_points3D_bin(path_out: str, xyz: np.ndarray, rgb_uint8: np.ndarray, errors: Optional[np.ndarray]=None):
    N = xyz.shape[0]
    if errors is None:
        errors = np.zeros((N,), dtype=np.float64)
    with open(path_out, "wb") as f:
        f.write(struct.pack("<Q", N))
        for i in range(N):
            f.write(struct.pack("<Q", i+1))
            f.write(struct.pack("<ddd", float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2])))
            f.write(struct.pack("BBB", int(rgb_uint8[i,0]), int(rgb_uint8[i,1]), int(rgb_uint8[i,2])))
            f.write(struct.pack("<d", float(errors[i])))
            f.write(struct.pack("<Q", 0))  # empty track


def choose_diverse_views(img_ids, flat_poses, K=6, seed=0):
    """
    Farthest-point sampling in pose space to get diverse views for density checks.
    Returns a list of image_ids.
    """
    rng = np.random.default_rng(seed)
    X = flat_poses.astype(np.float32)
    N = X.shape[0]
    first = int(np.argmax(np.linalg.norm(X - X.mean(0, keepdims=True), axis=1)))
    sel = [first]
    dist = np.linalg.norm(X - X[first], axis=1)
    for _ in range(1, min(K, N)):
        nxt = int(np.argmax(dist))
        sel.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(X - X[nxt], axis=1))
    return [img_ids[i] for i in sel]

def project_points(P, xyz):
    """
    P: 3x4; xyz: [N,3] float64
    returns uv: [N,2] float64, mask: [N] bool (z>0)
    """
    Xh = np.concatenate([xyz, np.ones((xyz.shape[0],1), dtype=np.float64)], axis=1)
    proj = (P @ Xh.T).T
    z = proj[:,2]
    mask = z > 1e-6
    proj[mask, :2] /= z[mask, None]
    uv = proj[:,:2]
    return uv, mask

def pixel_footprint_world(K, z):
    """
    Approximate size in world units that 1 pixel covers at depth z.
    Uses fx,fy from K; returns scalar (geometric mean of x/y footprints).
    """
    fx, fy = K[0,0], K[1,1]
    # footprint ~ z/f
    fx = max(fx, 1e-6); fy = max(fy, 1e-6)
    sx = z / fx; sy = z / fy
    return np.sqrt(max(sx*sy, 1e-12))

def thin_by_screenspace_density(
    xyz, rgb, cams_by_id, P_by, view_ids, target_ppp=0.75, image_sizes=None, seed=0
):
    """
    Blue-noise-ish thinning: ensure avg density across 'view_ids' is <= target_ppp.
    We compute per-pixel load via hashing and do probabilistic keep/drop.

    Returns keep_mask [N] bool.
    """
    if target_ppp <= 0 or len(view_ids) == 0:
        return np.ones((xyz.shape[0],), dtype=bool)

    rng = np.random.default_rng(seed)
    N = xyz.shape[0]
    keep = np.ones((N,), dtype=bool)

    # Accumulate per-point pressure across views
    pressure = np.zeros((N,), dtype=np.float32)
    counts   = np.zeros((N,), dtype=np.int32)

    # A small jittered hash for pixel bins to avoid aliasing
    def hash2(u, v, W, H):
        ui = np.clip(np.floor(u).astype(np.int64), 0, W-1)
        vi = np.clip(np.floor(v).astype(np.int64), 0, H-1)
        return (ui * 73856093) ^ (vi * 19349663)

    # Estimate per-view pixel density with stable hashing
    for vid in view_ids:
        cam = cams_by_id[vid]
        W, H = cam.width, cam.height if image_sizes is None else image_sizes.get(vid, (cam.width, cam.height))
        P = P_by[vid]
        uv, mask = project_points(P, xyz)
        u, v = uv[mask,0], uv[mask,1]
        inimg = (u>=0) & (u<W-1) & (v>=0) & (v<H-1)
        idx = np.nonzero(mask)[0][inimg]
        if idx.size == 0: 
            continue

        u, v = u[inimg], v[inimg]
        keys = hash2(u, v, W, H)
        order = np.argsort(keys)
        keys_sorted = keys[order]
        idx_sorted  = idx[order]

        # run-length segments per hashed pixel
        edge = np.nonzero(np.diff(keys_sorted))[0] + 1
        seg_starts = np.concatenate([[0], edge])
        seg_ends   = np.concatenate([edge, [keys_sorted.size]])

        # each segment -> points that land in same pixel bin
        for s, e in zip(seg_starts, seg_ends):
            pts = idx_sorted[s:e]
            k = e - s
            # desired count per pixel ~ target_ppp
            if k <= target_ppp:
                pressure[pts] += 0.0
            else:
                # excess load
                excess = k - target_ppp
                # assign higher pressure to low-gradient neighbors first? (not available here)
                pressure[pts] += excess / max(k, 1)
            counts[pts] += 1

    counts = np.maximum(counts, 1)
    avg_pressure = pressure / counts

    # Convert pressure to drop probability (smoothstep)
    # pressure=0 -> p_drop=0; pressure>=1 -> ~heavy drop
    p_drop = np.clip(0.5 * avg_pressure, 0.0, 0.95)
    draw = rng.random(N)
    keep = draw > p_drop
    # Always keep a minimal random subset if we got too aggressive
    return keep

def merge_covariance_aware(
    xyz, rgb, K_ref, R_ref, radius_mult=0.75, min_cluster=1, seed=0
):
    """
    Merge near-duplicate points using a radius derived from pixel footprint at depth.
    Keeps edges: we compute a per-point radius, then do grid hashing with that local radius.

    Returns merged_xyz, merged_rgb
    """
    if radius_mult <= 0:
        return xyz, rgb
    N = xyz.shape[0]
    rng = np.random.default_rng(seed)

    # Per-point adaptive radius
    # estimate depth along camera forward axis ~ R_ref[2] dotted with (X - C)
    # If R_ref is world->cam, camera forward in world is R_ref.T[:,2] *(-1) depending on convention.
    # Use z = |X| as crude proxy if pose not provided.
    if R_ref is None:
        z = np.linalg.norm(xyz, axis=1)
    else:
        # approximate z by projecting onto some canonical forward dir
        forward = R_ref.T[:,2]
        z = np.abs(xyz @ forward)

    base = np.array([[K_ref[0,0], 0, K_ref[0,2]],
                     [0, K_ref[1,1], K_ref[1,2]],
                     [0, 0, 1]], dtype=np.float64)
    px = np.array([1.0, 1.0, 1.0], dtype=np.float64)  # 1px
    # footprint at depth z
    fpt = np.array([pixel_footprint_world(K_ref, zi) for zi in z], dtype=np.float64)
    radii = np.clip(radius_mult * fpt, 1e-4, np.percentile(fpt, 90))  # clamp extremes

    # Hash into an adaptive grid by normalizing with radii
    keyx = np.floor(xyz[:,0] / radii).astype(np.int64)
    keyy = np.floor(xyz[:,1] / radii).astype(np.int64)
    keyz = np.floor(xyz[:,2] / radii).astype(np.int64)
    keys = keyx * 73856093 ^ keyy * 19349663 ^ keyz * 83492791
    order = np.argsort(keys)
    keys_sorted = keys[order]
    xyz_s = xyz[order]; rgb_s = rgb[order]; r_s = radii[order]

    edge = np.nonzero(np.diff(keys_sorted))[0] + 1
    boundaries = np.concatenate([[0], edge, [len(keys_sorted)]])

    out_xyz, out_rgb = [], []
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        seg_xyz = xyz_s[s:e]; seg_rgb = rgb_s[s:e]; seg_r = r_s[s:e]
        if e - s <= min_cluster:
            out_xyz.append(seg_xyz.mean(0))
            out_rgb.append(seg_rgb.mean(0))
            continue
        # Within the bin, do a small-radius mean-shift style merge:
        # greedy: pop a seed, absorb neighbors within max(seg_r)
        taken = np.zeros((e - s,), dtype=bool)
        for i in range(e - s):
            if taken[i]: continue
            center = seg_xyz[i]
            rad = seg_r[i]
            d2 = np.sum((seg_xyz - center)**2, axis=1)
            group = d2 <= (rad*rad)
            taken |= group
            out_xyz.append(seg_xyz[group].mean(0))
            out_rgb.append(seg_rgb[group].mean(0))

    return np.asarray(out_xyz), np.asarray(out_rgb)


# ==========================
# Main pipeline
# ==========================

def dense_init(args):
    scene_root = os.path.abspath(args.scene_root)
    sparse_dir = os.path.join(scene_root, "sparse", "0")
    images_dir = image_dir(scene_root, args.images_subdir)

    log(f"Scene   : {scene_root}")
    log(f"Sparse  : {sparse_dir}")
    log(f"Images  : {images_dir}")

    rec, cams, imgs = load_reconstruction(sparse_dir)
    img_ids = sorted(list(imgs.keys()))
    log(f"Loaded {len(cams)} cameras, {len(imgs)} images.")

    # Build camera dicts
    K_by: Dict[int,np.ndarray] = {}
    R_by: Dict[int,np.ndarray] = {}
    t_by: Dict[int,np.ndarray] = {}
    P_by: Dict[int,np.ndarray] = {}
    C_by: Dict[int,np.ndarray] = {}
    name_by: Dict[int,str] = {}
    size_by: Dict[int,Tuple[int,int]] = {}  # (w,h) from camera intrinsics

    flat_poses = []
    for iid in img_ids:
        im = imgs[iid]
        cam = cams[im.camera_id]
        K = K_from_camera(cam)
        R, t = pose_world2cam(im)
        P = P_from_KRt(K,R,t)
        C = cam_center_world(R,t)
        K_by[iid], R_by[iid], t_by[iid], P_by[iid], C_by[iid] = K, R, t, P, C
        name_by[iid] = im.name
        size_by[iid] = (cam.width, cam.height)
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = t.reshape(3)
        flat_poses.append(T.reshape(-1))

    flat_poses = np.stack(flat_poses, axis=0)
    # Reference selection + neighbors
    num_refs = int(round(args.num_refs * len(img_ids)))
    try:
        log("Selecting reference views by sparse point visibility...")
        refs = select_cameras_by_visibility(rec, num_refs)
        # Get local indices for the nn_table
        img_id_to_local_idx = {iid: i for i, iid in enumerate(img_ids)}
        refs_local = [img_id_to_local_idx[i] for i in refs]
    except (ValueError, KeyError) as e:
        log(f"Visibility selection failed ({e}), falling back to k-means on poses.")
        refs_local = select_cameras_kmeans(flat_poses, num_refs)
        refs = [img_ids[i] for i in refs_local]

    nn_table = nearest_neighbors(flat_poses, max(1, args.nns_per_ref))  # local indices

    # Matcher
    if not _ROMA_OK:
        raise RuntimeError("romatch not available. Please `pip install romatch` and retry.")
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    matcher = RomaMatcher(device=device, mode=args.roma_model)
    log(f"RoMa: {args.roma_model} on {device}")

    rng = np.random.default_rng(args.seed)

    # Viz input sparse point cloud if requested
    if args.viz and rec.points3D:
        pts, cols = [], []
        for p in rec.points3D.values():
            pts.append(np.asarray(p.xyz, dtype=np.float64))
            cols.append(np.asarray(p.color, dtype=np.uint8))
        if len(pts):
            xyz0 = np.stack(pts, axis=0)
            rgb0 = np.stack(cols, axis=0)
            if _HAS_O3D:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz0)
                pcd.colors = o3d.utility.Vector3dVector((rgb0.astype(np.float32)/255.0))
                o3d.visualization.draw_geometries([pcd], window_name="Input Sparse (COLMAP)")

    all_xyz, all_rgb, all_err = [], [], []
    t0 = time.time()

    # Process references
    for ref_local in tqdm(range(num_refs), desc="EDGS init"):
        ref_id = img_ids[ref_local]
        if ref_id not in refs:
            continue

        ref_name = name_by[ref_id]
        ref_path = find_image(images_dir, ref_name)
        imA = Image.open(ref_path).convert("RGB")
        wA_img, hA_img = imA.size
        wA_cam, hA_cam = size_by[ref_id]
        w_match, h_match = matcher.model.w_resized, matcher.model.h_resized

        # Collect K-NN flows for multi-view aggregation
        local_nns = nn_table[ref_local][:args.nns_per_ref]
        if len(local_nns) == 0: continue

        warp_list, weight_list, nn_ids = [], [], []
        for nn_local in local_nns:
            nbr_id = img_ids[nn_local]
            if nbr_id == ref_id: continue
            imB = Image.open(find_image(images_dir, name_by[nbr_id])).convert("RGB")
            
            # Calculate parallax for weighting
            parallax_rad = parallax_angle_for_cameras(R_by[ref_id], R_by[nbr_id])

            # Perform robust, cycle-consistent matching
            warp_hw4, weight_hw = match_and_filter_pair(
                matcher, imA, imB, parallax_rad, cycle_thresh_px=1.5
            )
            
            if warp_hw4 is not None:
                warp_list.append(warp_hw4)
                weight_list.append(weight_hw)
                nn_ids.append(nbr_id)

        if not weight_list:
            continue

        # --- Multi-view aggregation and sampling ---
        TOP_K_TRACKS = 3 # Triangulate with top 3 valid neighbors per pixel
        H, W = weight_list[0].shape
        
        weight_stack = torch.stack(weight_list, dim=0) # [Num_Neighbors, H, W]
        warp_stack = torch.stack(warp_list, dim=0)     # [Num_Neighbors, H, W, 4]

        # Get top-k weights and their indices (which correspond to the neighbor)
        top_weights, top_k_indices = torch.topk(
            weight_stack, k=min(TOP_K_TRACKS, len(nn_ids)), dim=0
        ) # [K, H, W]

        # Use the best weight for each pixel as the basis for sampling
        best_weight_map, _ = torch.max(weight_stack, dim=0) # [H, W]
        best_weight_flat = best_weight_map.reshape(-1)      # [H*W]

        # EDGS certainty cap then multinomial sampling + coverage
        cert_np = best_weight_flat.detach().cpu().numpy()
        # Cap is not strictly needed as weights are parallax-aware, but can help
        # cap = getattr(matcher, "sample_thresh", 0.9)
        # cert_np = np.minimum(cert_np, cap)
        if cert_np.max() < 1e-6: continue
        cert_np = cert_np / (cert_np.max() + 1e-12)

        # Get coordinates in reference image A for all potential pixels
        # Note: all warps in the stack share the same reference coordinates (xA, yA)
        coords_A_flat = warp_stack[0, ..., :2].reshape(-1, 2) # [H*W, 2]
        xA = (coords_A_flat[:,0].cpu().numpy() + 1.0) * 0.5 * (w_match - 1)
        yA = (coords_A_flat[:,1].cpu().numpy() + 1.0) * 0.5 * (h_match - 1)

        # keep inside safe border (avoid ~2px border)
        border = 2.0
        inside = (xA >= border) & (xA <= w_match-1-border) & (yA >= border) & (yA <= h_match-1-border)

        # certainty-weighted multinomial
        keep_weights = cert_np.copy()
        keep_weights[~inside] = 0.0
        if keep_weights.sum() == 0:
            continue
        m_main = int(args.matches_per_ref * 0.7)
        # Ensure we don't request more samples than available valid pixels
        num_valid_pixels = np.count_nonzero(keep_weights)
        if num_valid_pixels == 0: continue
        
        p_norm = keep_weights/keep_weights.sum()
        idx_main = rng.choice(
            keep_weights.size, 
            size=min(m_main, num_valid_pixels), 
            replace=False, 
            p=p_norm
        )

        # coverage fill: per-tile top-1 by certainty
        gx = np.floor(xA / max(1, w_match // 24)).astype(np.int32)
        gy = np.floor(yA / max(1, h_match // 24)).astype(np.int32)
        bins = gx * 100000 + gy
        order = np.argsort(-cert_np)  # descending
        seen = set(); idx_cov = []
        for ii in order:
            if not inside[ii]: continue
            b = bins[ii]
            if b in seen: continue
            seen.add(b); idx_cov.append(ii)
            if len(idx_cov) >= args.matches_per_ref - len(idx_main):
                break

        sel_idx = np.unique(np.concatenate([idx_main, np.array(idx_cov, dtype=np.int64)]))
        if sel_idx.size == 0:
            continue

        # --- Triangulation from Top-K Tracks ---
        
        # Get ref image coords (uvA) and colors for all S selected pixels
        xA_sel, yA_sel = xA[sel_idx], yA[sel_idx]
        
        # Colors from reference
        imA_np = np.asarray(imA, dtype=np.uint8)
        sxA_img, syA_img = wA_img / float(w_match), hA_img / float(h_match)
        xA_img, yA_img = xA_sel * sxA_img, yA_sel * syA_img
        # Bilinear sample
        xa0, ya0 = np.floor(xA_img).astype(np.int32), np.floor(yA_img).astype(np.int32)
        xa1, ya1 = xa0 + 1, ya0 + 1
        wa = (xa1 - xA_img)*(ya1 - yA_img)
        wb = (xA_img - xa0)*(ya1 - yA_img)
        wc = (xa1 - xA_img)*(yA_img - ya0)
        wd = (xA_img - xa0)*(yA_img - ya0)
        
        # Clip to prevent out-of-bounds access
        xa0, ya0 = np.clip(xa0, 0, wA_img-1), np.clip(ya0, 0, hA_img-1)
        xa1, ya1 = np.clip(xa1, 0, wA_img-1), np.clip(ya1, 0, hA_img-1)

        Ia, Ib = imA_np[ya0, xa0], imA_np[ya0, xa1]
        Ic, Id = imA_np[ya1, xa0], imA_np[ya1, xa1]
        rgb_ref = (Ia.astype(np.float32)*wa[:,None] + Ib.astype(np.float32)*wb[:,None] + Ic.astype(np.float32)*wc[:,None] + Id.astype(np.float32)*wd[:,None]) / 255.0

        # Scale to camera-sized pixels for triangulation
        sxA, syA = wA_cam / float(w_match), hA_cam / float(h_match)
        uvA = np.stack([xA_sel * sxA, yA_sel * syA], axis=1)  # [S, 2]

        # Now, iterate through the K tracks and triangulate for each
        for k_track in range(top_k_indices.shape[0]):
            # Get neighbor and warp data for this track level (k) for all selected pixels
            neighbor_indices_k = top_k_indices[k_track].reshape(-1)[sel_idx].cpu().numpy() # [S]
            
            # Gather the warps for this track level.
            # This requires selecting from warp_stack based on neighbor_indices_k.
            # `warp_stack` is [Num_Neighbors, H, W, 4]
            flat_warps = warp_stack.permute(1, 2, 0, 3).reshape(-1, len(nn_ids), 4) # [H*W, N, 4]
            sel_warps_all_neighbors = flat_warps[sel_idx] # [S, N, 4]
            
            # Use numpy to perform the advanced integer indexing
            # This selects the correct neighbor's warp for each of the S pixels
            row_indices = np.arange(sel_idx.shape[0])
            sel_warps_k = sel_warps_all_neighbors[row_indices, neighbor_indices_k].cpu().numpy() # [S, 4]

            # Extract B coordinates (normalized)
            xB_norm, yB_norm = sel_warps_k[:, 2], sel_warps_k[:, 3]

            # Group samples by neighbor to triangulate with correct P2 / K2 sizes
            groups: Dict[int, List[int]] = {}
            for i, neighbor_k_idx in enumerate(neighbor_indices_k):
                nbr_id = nn_ids[neighbor_k_idx]
                groups.setdefault(nbr_id, []).append(i)

            for nbr_id, idxs_in_group in groups.items():
                if not idxs_in_group: continue
                idxs = np.asarray(idxs_in_group, dtype=np.int64)
                
                # B image geometry
                wB_cam, hB_cam = size_by[nbr_id]
                sxB, syB = wB_cam / float(w_match), hB_cam / float(h_match)

                # Build uvB for just this neighbor group
                xB = (xB_norm[idxs] + 1.0) * 0.5 * (w_match - 1)
                yB = (yB_norm[idxs] + 1.0) * 0.5 * (h_match - 1)
                uvB = np.stack([xB * sxB, yB * syB], axis=1)

                # Pre-triangulation Sampson gating is not strictly necessary with
                # cycle consistency but can be kept as an extra check.
                if args.sampson_thresh > 0:
                    F = fundamental_from_world2cam(K_by[ref_id], R_by[ref_id], t_by[ref_id],
                                                   K_by[nbr_id], R_by[nbr_id], t_by[nbr_id])
                    se = sampson_error(F, uvA[idxs], uvB)
                    good = (se < float(args.sampson_thresh))
                    if not np.any(good): continue
                    
                    idxs = idxs[good]
                    uvB = uvB[good]
                
                if idxs.size == 0: continue

                P1, P2 = P_by[ref_id], P_by[nbr_id]
                Xi = dlt_triangulate_batch(P1, P2, uvA[idxs], uvB)

                err1 = reprojection_errors(P1, Xi, uvA[idxs])
                err2 = reprojection_errors(P2, Xi, uvB)
                err  = np.maximum(err1, err2)
                keep = (err <= float(args.reproj_thresh))
                
                keep &= cheirality_mask(P1, Xi)
                keep &= cheirality_mask(P2, Xi)
                if args.min_parallax_deg > 0:
                    keep &= parallax_mask(C_by[ref_id], C_by[nbr_id], Xi, min_deg=args.min_parallax_deg)
                
                if not np.any(keep): continue

                Xw = Xi[keep][:,:3].astype(np.float64)
                col = rgb_ref[idxs][keep].astype(np.float32)
                e   = err[keep].astype(np.float64)

                all_xyz.append(Xw)
                all_rgb.append(col)
                all_err.append(e)

    if not all_xyz:
        raise RuntimeError("No points triangulated. Try increasing --num_refs / --nns_per_ref / --matches_per_ref or lowering thresholds.")

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)

    if not all_xyz:
        raise RuntimeError("No points triangulated. Try increasing --num_refs / --nns_per_ref / --matches_per_ref or lowering thresholds.")

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    err = np.concatenate(all_err, axis=0)
    log(f"Triangulated points: {xyz.shape[0]} in {time.time()-t0:.1f}s.")

    # === New: density-aware thinning / merging ===
    if args.thin_ppp > 0 or args.merge_radius_mult > 0:
        # Pick diverse views to evaluate screen-space density
        diverse_views = choose_diverse_views(
            img_ids, flat_poses, K=max(1, args.thin_views), seed=args.seed
        )
        log(f"Density check over {len(diverse_views)} views.")

        # Build camera dict for the views
        cams_by_id = {iid: cams[imgs[iid].camera_id] for iid in diverse_views}

        # Screen-space thinning
        if args.thin_ppp > 0:
            keep_mask = thin_by_screenspace_density(
                xyz, rgb, cams_by_id, P_by, diverse_views,
                target_ppp=float(args.thin_ppp), image_sizes=None, seed=args.seed
            )
            # Enforce min_keep
            if keep_mask.sum() < args.min_keep:
                # keep best by error (lowest)
                order = np.argsort(err)
                keep_mask[:] = False
                keep_mask[order[:args.min_keep]] = True
            xyz, rgb, err = xyz[keep_mask], rgb[keep_mask], err[keep_mask]
            log(f"Screen-space thinning ⇒ {xyz.shape[0]} points.")

        # Covariance-aware merge (uses reference intrinsics)
        if args.merge_radius_mult > 0:
            # Use the first diverse view as reference for K_ref/R_ref
            ref_id_merge = diverse_views[0]
            K_ref = K_by[ref_id_merge]
            R_ref = R_by[ref_id_merge]
            xyz, rgb = merge_covariance_aware(
                xyz, rgb, K_ref, R_ref,
                radius_mult=float(args.merge_radius_mult),
                min_cluster=1, seed=args.seed
            )
            err = np.zeros((xyz.shape[0],), dtype=np.float64)  # reset (conservative)
            log(f"Covariance-aware merge ⇒ {xyz.shape[0]} points.")

        # Ensure floor
        if args.min_keep > 0 and xyz.shape[0] < args.min_keep:
            log(f"Padding back up to min_keep={args.min_keep} by random re-adds (no-op here).")

    # Optional voxel DS (keep small if used)
    if args.voxel_size > 0:
        xyz, rgb = voxel_downsample(xyz, rgb, args.voxel_size)
        err = np.zeros((xyz.shape[0],), dtype=np.float64)
        log(f"Voxel downsampled to {xyz.shape[0]} (voxel={args.voxel_size}).")

    # (Optional) clamp count
    if args.max_points > 0 and xyz.shape[0] > args.max_points:
        sel = np.random.default_rng(args.seed).choice(xyz.shape[0], size=args.max_points, replace=False)
        xyz, rgb, err = xyz[sel], rgb[sel], err[sel]
        log(f"Capped to {xyz.shape[0]} points (--max_points).")

    # Write COLMAP points3D.bin
    out_path = os.path.join(sparse_dir, args.out_name)
    write_points3D_bin(out_path, xyz, to_uint8_rgb(rgb), errors=err)
    log(f"Wrote dense COLMAP: {out_path}")

    if args.viz and _HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector((rgb.astype(np.float32)))
        o3d.visualization.draw_geometries([pcd], window_name="Dense init (EDGS-like)")


# ==========================
# CLI
# ==========================

def build_argparser():
    ap = argparse.ArgumentParser("EDGS-like Dense Initialization (RoMa + triangulation)")
    ap.add_argument("--scene_root", type=str, required=True, help="Path containing images*/ and sparse/0/")
    ap.add_argument("--images_subdir", type=str, default="images_2", help="Which images dir to read")
    ap.add_argument("--out_name", type=str, default="points3D_dense.bin", help="Output filename under sparse/0/")
    ap.add_argument("--roma_model", type=str, default="outdoor", choices=["outdoor","indoor"], help="RoMa model variant")
    ap.add_argument("--cpu", action="store_true", help="Force CPU for RoMa (slow)")
    # EDGS knobs
    ap.add_argument("--num_refs", type=float, default=0.8, help="Fraction of frames to use as references (e.g., 0.8 for 80%%)")
    ap.add_argument("--nns_per_ref", type=int, default=3, help="Nearest neighbors per ref (use 3-5 for robustness)")
    ap.add_argument("--matches_per_ref", type=int, default=15000, help="Samples per ref after aggregation & threshold")
    ap.add_argument("--certainty_thresh", type=float, default=0.2, help="Min certainty to consider a pixel")
    ap.add_argument("--reproj_thresh", type=float, default=3.0, help="Max reprojection error (px)")
    ap.add_argument("--sampson_thresh", type=float, default=6.0, help="Max Sampson error (px^2) before triangulation (<=0 to disable)")
    ap.add_argument("--min_parallax_deg", type=float, default=0.1, help="Min parallax (deg); set 0 to disable")
    # Density-aware pruning/merging
    ap.add_argument("--thin_ppp", type=float, default=0.75,
                    help="Target avg points-per-pixel across sampled views (0 disables).")
    ap.add_argument("--thin_views", type=int, default=6,
                    help="Number of diverse views for screen-space density estimation.")
    ap.add_argument("--merge_radius_mult", type=float, default=0.75,
                    help="Multiplier converting 1px footprint at depth to 3D merge radius (0 disables).")
    ap.add_argument("--min_keep", type=int, default=200000,
                    help="Never prune below this many points (safety).")
    # Output shaping
    ap.add_argument("--voxel_size", type=float, default=0.0, help="Optional voxel size (scene units); keep small or 0")
    ap.add_argument("--max_points", type=int, default=3500000, help="Optional cap on total points (0 = unlimited)")
    ap.add_argument("--viz", action="store_true", help="Visualize input/output point clouds (Open3D)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    dense_init(args)
