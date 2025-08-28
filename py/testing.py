#!/usr/bin/env python3
#Garden 26.9
"""
Dense, accurate COLMAP pointcloud initializer (EDGS-inspired, RoMa-driven)

Goal
-----
Take an existing COLMAP reconstruction (poses + intrinsics + images) and build a
dense-but-accurate point cloud to seed a 3DGS densification stage.

What it does (in short)
-----------------------
1) Loads COLMAP cameras/poses from scene_root/sparse/0 and images from images_*.
2) Picks reference views (visibility-aware if a sparse cloud exists; else k-centers on poses).
3) For each reference, finds pose-nearest neighbors.
4) Runs RoMa dense matching to get (xA,yA)↔(xB,yB) at grid resolution + per-pixel certainty.
5) Aggregates neighbors with per-pixel argmax certainty.
6) EDGS-style sampling: cap certainty → multinomial sampling + coverage-grid fill.
7) Geometry gating:
   - optional Sampson epipolar error (fast pre-triangulation gate)
   - triangulation (DLT)
   - reprojection error in both views
   - cheirality in both views
   - minimum parallax (deg)
8) Colors are taken from the reference image (bilinear).
9) Writes:
   - COLMAP-compatible points3D_dense.bin (under sparse/0)
   - Optional seeds .npz (xyz float32, rgb float32 in [0,1]) for GS bootstrapping.

Usage
-----
python dense_colmap_init.py \
  --scene_root /path/to/scene \
  --images_subdir images_4 \
  --num_refs 0.8 --nns_per_ref 4 --matches_per_ref 20000 \
  --roma_model outdoor --reproj_thresh 2.5 --sampson_thresh 6.0 --min_parallax_deg 0.5 \
  --out_name points3D_dense.bin --write_npz --npz_name seeds_dense.npz --viz

Dependencies
------------
pip install pycolmap romatch Pillow numpy scipy tqdm open3d  # (open3d optional for --viz)

Notes
-----
- This is self-contained and only needs a valid COLMAP workspace:
     scene_root/
       images*/               (images or images_2/images_4/images_8)
       sparse/0/              (cameras.bin, images.bin, points3D.bin)
- If your images are elsewhere, set --images_subdir accordingly.
"""

import os, argparse, struct, time
from typing import Dict, Tuple, List, Optional

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

def log(s): print(f"[dense-init] {s}")

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
    K = np.eye(3, dtype=np.float64)
    model = str(cam.model.name).upper()
    p = np.asarray(cam.params, dtype=np.float64)
    w, h = cam.width, cam.height
    if "PINHOLE" in model and "SIMPLE" not in model:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif "SIMPLE_PINHOLE" in model:
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif "SIMPLE_RADIAL" in model or model == "RADIAL":
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif "OPENCV" in model or "FISHEYE" in model:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
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
    if not rec.points3D:
        raise ValueError("Visibility-based selection requires a sparse point cloud.")
    img_to_pts = {
        img.image_id: [p.point3D_id for p in img.points2D if p.has_point3D() and p.point3D_id != -1]
        for img in rec.images.values()
    }
    K = min(K, len(img_to_pts))
    selected_cams = []
    covered_pts = set()
    scores = {iid: len(pids) for iid, pids in img_to_pts.items()}
    for _ in range(K):
        if not scores: break
        best_cam = max(scores, key=scores.get)
        selected_cams.append(best_cam)
        newly_covered = set(img_to_pts[best_cam]) - covered_pts
        covered_pts.update(newly_covered)
        del scores[best_cam]
        for cam_id, cam_pts in img_to_pts.items():
            if cam_id in scores:
                scores[cam_id] = len(set(cam_pts) - covered_pts)
    return sorted(selected_cams)

def select_cameras_kcenters(flat_poses: np.ndarray, K: int) -> List[int]:
    X = np.asarray(flat_poses, dtype=np.float32)
    N = X.shape[0]; K = max(1, min(int(K), N))
    mu = X.mean(axis=0, keepdims=True); sigma = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - mu) / sigma
    first = int(np.argmax(np.einsum("nd,nd->n", Xn, Xn)))
    centers = [first]
    dist = np.linalg.norm(Xn - Xn[first], axis=1); dist[first] = -np.inf
    for _ in range(1, K):
        nxt = int(np.argmax(dist)); centers.append(nxt)
        d = np.linalg.norm(Xn - Xn[nxt], axis=1)
        dist = np.minimum(dist, d); dist[nxt] = -np.inf
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
        assert _ROMA_OK, "romatch not available. Please `pip install romatch` and retry."
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
        flow_or_warp, cert = self.model.match(imA, imB, device=self.device)
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

def parallax_mask(C1: np.ndarray, C2: np.ndarray, X: np.ndarray, min_deg=0.5) -> np.ndarray:
    v1 = X[:,:3] - C1.reshape(1,3)
    v2 = X[:,:3] - C2.reshape(1,3)
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True) + 1e-12
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True) + 1e-12
    ang = np.degrees(np.arccos(np.clip(np.sum(v1*v2,axis=1), -1.0, 1.0)))
    return ang >= float(min_deg)

def fundamental_from_world2cam(K1,R1,t1,K2,R2,t2) -> np.ndarray:
    R = R2 @ R1.T
    t = (t2 - R @ t1).reshape(3)
    E = skew(t) @ R
    K1i = np.linalg.inv(K1); K2i = np.linalg.inv(K2)
    F = K2i.T @ E @ K1i
    return F

def sampson_error(F: np.ndarray, uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    N = uv1.shape[0]
    x1 = np.concatenate([uv1, np.ones((N,1))], axis=1)
    x2 = np.concatenate([uv2, np.ones((N,1))], axis=1)
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    x2Fx1 = np.sum(x2 * Fx1, axis=1)
    den = Fx1[:,0]**2 + Fx1[:,1]**2 + Ftx2[:,0]**2 + Ftx2[:,1]**2 + 1e-12
    return (x2Fx1**2) / den


# ==========================
# EDGS-style sampling (cap + multinomial + coverage fill)
# ==========================

def select_samples_with_coverage(cert_map: torch.Tensor, M: int, cap: float = 0.9,
                                 border: int = 2, tiles: int = 24) -> np.ndarray:
    cert = cert_map.clone()
    cert = torch.clamp(cert, max=cap)
    H, W = cert.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=cert.device), torch.arange(W, device=cert.device), indexing="ij")
    inside = (xx >= border) & (xx <= W - 1 - border) & (yy >= border) & (yy <= H - 1 - border)
    weights = (cert * inside.float()).reshape(-1)
    s = weights.sum()
    if s <= 0:
        return np.zeros((0,), dtype=np.int64)
    weights = (weights / s).detach().cpu().numpy()

    m_main = int(M * 0.7)
    idx_main = np.random.choice(weights.size, size=min(m_main, weights.size), replace=False, p=weights)

    # coverage fill (tile top-1)
    tile = max(1, W // tiles)
    gx = (xx // tile).reshape(-1).cpu().numpy()
    gy = (yy // tile).reshape(-1).cpu().numpy()
    bins = gx * 100000 + gy
    order = np.argsort(-weights)  # descending by weight
    seen = set(); idx_cov = []
    for i in order:
        if weights[i] <= 0: break
        b = int(bins[i])
        if b in seen: continue
        seen.add(b); idx_cov.append(i)
        if len(idx_cov) >= M - len(idx_main): break

    sel_idx = np.unique(np.concatenate([idx_main, np.asarray(idx_cov, dtype=np.int64)]))
    return sel_idx


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
# Writers
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

def write_npz(path_out: str, xyz: np.ndarray, rgb01: np.ndarray):
    np.savez_compressed(path_out, xyz=xyz.astype(np.float32), rgb=rgb01.astype(np.float32))


# ==========================
# Main pipeline
# ==========================

def dense_init(args):
    np.random.seed(args.seed)
    if not _ROMA_OK:
        raise RuntimeError("romatch not available. Please `pip install romatch` and retry.")

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
    num_refs = int(round(args.num_refs * len(img_ids))) if args.num_refs <= 1.0 else int(args.num_refs)
    try:
        log("Selecting reference views by sparse visibility...")
        refs = select_cameras_by_visibility(rec, num_refs)
        img_id_to_local_idx = {iid: i for i, iid in enumerate(img_ids)}
        refs_local = [img_id_to_local_idx[i] for i in refs]
    except (ValueError, KeyError) as e:
        log(f"Visibility selection failed ({e}), using k-centers on poses.")
        refs_local = select_cameras_kcenters(flat_poses, num_refs)
        refs = [img_ids[i] for i in refs_local]

    nn_table = nearest_neighbors(flat_poses, max(1, args.nns_per_ref))  # local indices

    # RoMa matcher
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    matcher = RomaMatcher(device=device, mode=args.roma_model)
    log(f"RoMa: {args.roma_model} on {device}")

    all_xyz, all_rgb, all_err = [], [], []
    t0 = time.time()

    for ref_local in tqdm(range(len(img_ids)), desc="Dense init"):
        ref_id = img_ids[ref_local]
        if ref_id not in refs:
            continue

        ref_name = name_by[ref_id]
        ref_path = find_image(images_dir, ref_name)
        imA = Image.open(ref_path).convert("RGB")
        wA_img, hA_img = imA.size
        wA_cam, hA_cam = size_by[ref_id]
        w_match, h_match = matcher.model.w_resized, matcher.model.h_resized

        # Collect K-NN flows for argmax aggregation
        local_nns = nn_table[ref_local][:args.nns_per_ref]
        if len(local_nns) == 0: continue

        warp_list, cert_list, nn_ids = [], [], []
        for nn_local in local_nns:
            nbr_id = img_ids[nn_local]
            if nbr_id == ref_id: continue
            imB = Image.open(find_image(images_dir, name_by[nbr_id])).convert("RGB")
            warp_hw, cert_hw = matcher.match_grids(imA, imB)  # [H,W,4], [H,W]
            cert_hw = torch.clamp(cert_hw, min=args.certainty_thresh)  # mild floor
            warp_list.append(warp_hw)
            cert_list.append(cert_hw)
            nn_ids.append(nbr_id)

        H, W = cert_list[0].shape
        cert_stack = torch.stack(cert_list, dim=0)                 # [K,H,W]
        best_cert, best_k = torch.max(cert_stack, dim=0)           # [H,W], [H,W]
        warp_stack = torch.stack(warp_list, dim=0)                 # [K,H,W,4]
        agg = warp_stack[best_k, torch.arange(H, device=device).unsqueeze(1), torch.arange(W, device=device)]  # [H,W,4]
        agg = agg.reshape(-1, 4)                                   # (H*W,4)

        # EDGS sampling (cap + multinomial + coverage)
        sel_idx = select_samples_with_coverage(best_cert, args.matches_per_ref, cap=matcher.sample_thresh,
                                               border=2, tiles=24)
        if sel_idx.size == 0:
            continue

        # Selected warps / winner-NN for each sample
        nn_idx_flat = best_k.reshape(-1).detach().cpu().numpy()[sel_idx]   # [S]
        sel = agg.detach().cpu().numpy()[sel_idx]                          # [S,4]
        xA = (sel[:,0] + 1.0)*0.5*(w_match-1)
        yA = (sel[:,1] + 1.0)*0.5*(h_match-1)
        xB_norm = sel[:,2]; yB_norm = sel[:,3]  # in [-1,1]

        # Colors from reference (bilinear)
        imA_np = np.asarray(imA, dtype=np.uint8)
        sxA_img = wA_img / float(w_match)
        syA_img = hA_img / float(h_match)
        xA_img = xA * sxA_img
        yA_img = yA * syA_img
        xa0 = np.clip(np.floor(xA_img).astype(np.int32), 0, wA_img-1)
        ya0 = np.clip(np.floor(yA_img).astype(np.int32), 0, hA_img-1)
        xa1 = np.clip(xa0+1, 0, wA_img-1)
        ya1 = np.clip(ya0+1, 0, hA_img-1)
        wa = (xa1 - xA_img)*(ya1 - yA_img)
        wb = (xA_img - xa0)*(ya1 - yA_img)
        wc = (xa1 - xA_img)*(yA_img - ya0)
        wd = (xA_img - xa0)*(yA_img - ya0)
        Ia = imA_np[ya0, xa0].astype(np.float32)
        Ib = imA_np[ya0, xa1].astype(np.float32)
        Ic = imA_np[ya1, xa0].astype(np.float32)
        Id = imA_np[ya1, xa1].astype(np.float32)
        rgb_ref = (Ia*wa[:,None] + Ib*wb[:,None] + Ic*wc[:,None] + Id*wd[:,None]) / 255.0  # [S,3]

        # Scale A to camera intrinsics resolution for triangulation
        sxA = wA_cam / float(w_match)
        syA = hA_cam / float(h_match)
        uvA = np.stack([xA * sxA, yA * syA], axis=1)  # [S,2]

        # Group by neighbor for correct B sizing
        groups: Dict[int, List[int]] = {}
        for i,(kidx) in enumerate(nn_idx_flat):
            nbr_id = nn_ids[kidx]
            groups.setdefault(nbr_id, []).append(i)

        for nbr_id, idxs in groups.items():
            idxs = np.asarray(idxs, dtype=np.int64)
            wB_cam, hB_cam = size_by[nbr_id]
            sxB = wB_cam / float(w_match)
            syB = hB_cam / float(h_match)

            xB = (xB_norm[idxs] + 1.0)*0.5*(w_match-1)
            yB = (yB_norm[idxs] + 1.0)*0.5*(h_match-1)
            uvB = np.stack([xB * sxB, yB * syB], axis=1)

            # Pre-triangulation Sampson (optional)
            if args.sampson_thresh > 0:
                F = fundamental_from_world2cam(K_by[ref_id], R_by[ref_id], t_by[ref_id],
                                               K_by[nbr_id], R_by[nbr_id], t_by[nbr_id])
                se = sampson_error(F, uvA[idxs], uvB)
                good = (se < float(args.sampson_thresh))
                if not np.any(good):
                    continue
                idxs = idxs[good]
                xB = xB[good]; yB = yB[good]
                uvB = uvB[good]
            if idxs.size == 0:
                continue

            P1, P2 = P_by[ref_id], P_by[nbr_id]
            Xi = dlt_triangulate_batch(P1, P2, uvA[idxs], uvB)     # [Mi,4]

            err1 = reprojection_errors(P1, Xi, uvA[idxs])
            err2 = reprojection_errors(P2, Xi, uvB)
            err  = np.maximum(err1, err2)
            keep = (err <= float(args.reproj_thresh))

            # cheirality + optional parallax
            keep &= cheirality_mask(P1, Xi)
            keep &= cheirality_mask(P2, Xi)
            if args.min_parallax_deg > 0:
                keep &= parallax_mask(C_by[ref_id], C_by[nbr_id], Xi, min_deg=args.min_parallax_deg)

            if not np.any(keep):
                continue

            Xw = Xi[keep][:,:3].astype(np.float64)
            col = rgb_ref[idxs][keep].astype(np.float32)
            e   = err[keep].astype(np.float64)

            all_xyz.append(Xw); all_rgb.append(col); all_err.append(e)

    if not all_xyz:
        raise RuntimeError("No points triangulated. Try increasing --num_refs / --nns_per_ref / --matches_per_ref or lowering thresholds.")

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    err = np.concatenate(all_err, axis=0)
    log(f"Triangulated points: {xyz.shape[0]} in {time.time()-t0:.1f}s.")

    # Optional voxel DS
    if args.voxel_size > 0:
        xyz, rgb = voxel_downsample(xyz, rgb, args.voxel_size)
        err = np.zeros((xyz.shape[0],), dtype=np.float64)  # reset errors conservatively
        log(f"Voxel downsampled to {xyz.shape[0]} (voxel={args.voxel_size}).")

    # Optional cap
    if args.max_points > 0 and xyz.shape[0] > args.max_points:
        sel = np.random.default_rng(args.seed).choice(xyz.shape[0], size=args.max_points, replace=False)
        xyz, rgb, err = xyz[sel], rgb[sel], err[sel]
        log(f"Capped to {xyz.shape[0]} points (--max_points).")

    # Write COLMAP points3D.bin
    out_path = os.path.join(sparse_dir, args.out_name)
    write_points3D_bin(out_path, xyz, to_uint8_rgb(rgb), errors=err)
    log(f"Wrote dense COLMAP: {out_path}")

    # Optional seeds for 3DGS
    if args.write_npz:
        npz_path = os.path.join(scene_root, args.npz_name)
        write_npz(npz_path, xyz, rgb)
        log(f"Wrote GS seeds: {npz_path}  (xyz float32, rgb float32 [0,1])")

    # Visualization
    if args.viz and _HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector((rgb.astype(np.float32)))
        o3d.visualization.draw_geometries([pcd], window_name="Dense init (EDGS-like)")

    return 0


# ==========================
# CLI
# ==========================

def build_argparser():
    ap = argparse.ArgumentParser("Dense COLMAP initializer (EDGS-style + RoMa)")
    ap.add_argument("--scene_root", type=str, required=True, help="Path containing images*/ and sparse/0/")
    ap.add_argument("--images_subdir", type=str, default="images_2", help="Which images dir to read under scene_root")
    ap.add_argument("--out_name", type=str, default="points3D_dense.bin", help="Output filename under sparse/0/")
    ap.add_argument("--roma_model", type=str, default="outdoor", choices=["outdoor","indoor"], help="RoMa model variant")
    ap.add_argument("--cpu", action="store_true", help="Force CPU for RoMa (slow)")

    # Selection
    ap.add_argument("--num_refs", type=float, default=0.8, help="Fraction (<=1) or count (>1) of frames to use as references")
    ap.add_argument("--nns_per_ref", type=int, default=4, help="Nearest neighbors per reference (3-5 is robust)")

    # Sampling / gating
    ap.add_argument("--matches_per_ref", type=int, default=15000, help="Samples per ref after aggregation")
    ap.add_argument("--certainty_thresh", type=float, default=0.15, help="Min certainty floor before selection")
    ap.add_argument("--reproj_thresh", type=float, default=2.5, help="Max reprojection error (px)")
    ap.add_argument("--sampson_thresh", type=float, default=6.0, help="Max Sampson error (px^2) pre-triangulation (<=0 to disable)")
    ap.add_argument("--min_parallax_deg", type=float, default=0.5, help="Min parallax (deg); set 0 to disable")

    # Output shaping
    ap.add_argument("--voxel_size", type=float, default=0.0, help="Optional voxel size (scene units); small or 0")
    ap.add_argument("--max_points", type=int, default=0, help="Optional cap on total points (0 = unlimited)")
    ap.add_argument("--write_npz", action="store_true", help="Also write seeds .npz (xyz,rgb) for GS bootstrapping")
    ap.add_argument("--npz_name", type=str, default="seeds_dense.npz", help="Name of the .npz seeds file (under scene_root)")
    ap.add_argument("--viz", action="store_true", help="Visualize output point cloud (Open3D)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    raise SystemExit(dense_init(args))
