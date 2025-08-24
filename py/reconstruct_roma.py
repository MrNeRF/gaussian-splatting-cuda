#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    num_refs = int(len(img_ids) * 0.8) # 40% of images 
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
        wA_match, hA_match = imA.size
        wA_cam, hA_cam = size_by[ref_id]

        # Collect K-NN flows for argmax aggregation
        local_nns = nn_table[ref_local][:args.nns_per_ref]
        if len(local_nns) == 0: continue

        warp_list, cert_list, nn_ids = [], [], []
        for nn_local in local_nns:
            nbr_id = img_ids[nn_local]
            if nbr_id == ref_id: continue
            imB = Image.open(find_image(images_dir, name_by[nbr_id])).convert("RGB")
            warp_hw4, cert_hw = matcher.match_grids(imA, imB)  # [H,W,4], [H,W]
            warp_list.append(warp_hw4)
            cert_list.append(cert_hw)
            nn_ids.append(nbr_id)

        # Stack neighbors and take per-pixel argmax certainty
        # Shapes: list K x [H,W,4] / K x [H,W]
        H, W = cert_list[0].shape
        cert_stack = torch.stack(cert_list, dim=0)            # [K,H,W]
        best_cert, best_k = torch.max(cert_stack, dim=0)      # [H,W], [H,W] (indices)
        warp_stack = torch.stack(warp_list, dim=0)            # [K,H,W,4]
        agg = warp_stack[best_k, torch.arange(H, device=device).unsqueeze(1), torch.arange(W, device=device)]  # [H,W,4]
        # Flatten for sampling
        agg = agg.reshape(-1, 4)                               # (H*W,4)
        best_cert = best_cert.reshape(-1)                      # (H*W,)
        best_k = best_k.reshape(-1)                            # (H*W,)

        # EDGS certainty cap then multinomial sampling + coverage
        cert_np = best_cert.detach().cpu().numpy()
        cap = getattr(matcher, "sample_thresh", 0.9)
        cert_np = np.minimum(cert_np, cap)                     # cap to 1 after normalization below
        # Avoid zero-sum
        cert_np = cert_np / (cert_np.max() + 1e-12)

        # coverage-aware grid bins on A (normalized -> pixel @ match size)
        xA = (agg[:,0].cpu().numpy() + 1.0) * 0.5 * (wA_match - 1)
        yA = (agg[:,1].cpu().numpy() + 1.0) * 0.5 * (hA_match - 1)

        # keep inside safe border (avoid ~2px border)
        border = 2.0
        inside = (xA >= border) & (xA <= wA_match-1-border) & (yA >= border) & (yA <= hA_match-1-border)

        # certainty-weighted multinomial
        keep_weights = cert_np.copy()
        keep_weights[~inside] = 0.0
        if keep_weights.sum() == 0:
            continue
        m_main = int(args.matches_per_ref * 0.7)
        idx_main = rng.choice(keep_weights.size, size=min(m_main, keep_weights.size), replace=False, p=keep_weights/keep_weights.sum())

        # coverage fill: per-tile top-1 by certainty
        gx = np.floor(xA / max(1, wA_match // 24)).astype(np.int32)
        gy = np.floor(yA / max(1, hA_match // 24)).astype(np.int32)
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

        # Build per-sample neighbor id
        nn_idx_flat = best_k.detach().cpu().numpy()[sel_idx]   # [S]
        # Selected warps
        sel = agg.detach().cpu().numpy()[sel_idx]              # [S,4]
        # Convert to pixel coords (match sizes)
        xA = (sel[:,0] + 1.0)*0.5*(wA_match-1)
        yA = (sel[:,1] + 1.0)*0.5*(hA_match-1)
        xB_norm = sel[:,2]; yB_norm = sel[:,3]  # in [-1,1], but B size depends on neighbor

        # Colors from reference
        imA_np = np.asarray(imA, dtype=np.uint8)
        # Bilinear sample
        xa0 = np.clip(np.floor(xA).astype(np.int32), 0, wA_match-1)
        ya0 = np.clip(np.floor(yA).astype(np.int32), 0, hA_match-1)
        xa1 = np.clip(xa0+1, 0, wA_match-1)
        ya1 = np.clip(ya0+1, 0, hA_match-1)
        wa = (xa1 - xA)*(ya1 - yA)
        wb = (xA - xa0)*(ya1 - yA)
        wc = (xa1 - xA)*(yA - ya0)
        wd = (xA - xa0)*(yA - ya0)
        Ia = imA_np[ya0, xa0].astype(np.float32)
        Ib = imA_np[ya0, xa1].astype(np.float32)
        Ic = imA_np[ya1, xa0].astype(np.float32)
        Id = imA_np[ya1, xa1].astype(np.float32)
        rgb_ref = (Ia*wa[:,None] + Ib*wb[:,None] + Ic*wc[:,None] + Id*wd[:,None]) / 255.0  # [S,3]

        # Scale to camera-sized pixels for triangulation (if match res != camera intrinsics res)
        sxA = wA_cam / float(wA_match)
        syA = hA_cam / float(hA_match)
        uvA = np.stack([xA * sxA, yA * syA], axis=1)  # [S,2]

        # Group samples by neighbor to triangulate with correct P2 / K2 sizes
        groups: Dict[int, List[int]] = {}
        for i,(kidx) in enumerate(nn_idx_flat):
            nbr_id = nn_ids[kidx]
            groups.setdefault(nbr_id, []).append(i)

        for nbr_id, idxs in groups.items():
            idxs = np.asarray(idxs, dtype=np.int64)
            # B image geometry
            nbr_name = name_by[nbr_id]
            imB = Image.open(find_image(images_dir, nbr_name)).convert("RGB")
            wB_match, hB_match = imB.size
            wB_cam, hB_cam = size_by[nbr_id]
            sxB = wB_cam / float(wB_match)
            syB = hB_cam / float(hB_match)

            # Build uvB for just this neighbor (from normalized xB/yB of the selected warps)
            xB = (xB_norm[idxs] + 1.0)*0.5*(wB_match-1)
            yB = (yB_norm[idxs] + 1.0)*0.5*(hB_match-1)
            uvB = np.stack([xB * sxB, yB * syB], axis=1)

            # Pre-triangulation Sampson gating (fast and effective)
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
                # also update colors subset if anything changed
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

            all_xyz.append(Xw)
            all_rgb.append(col)
            all_err.append(e)

    if not all_xyz:
        raise RuntimeError("No points triangulated. Try increasing --num_refs / --nns_per_ref / --matches_per_ref or lowering thresholds.")

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    err = np.concatenate(all_err, axis=0)
    log(f"Triangulated points: {xyz.shape[0]} in {time.time()-t0:.1f}s.")

    # Optional voxel DS (keep small if used)
    if args.voxel_size > 0:
        xyz, rgb = voxel_downsample(xyz, rgb, args.voxel_size)
        err = np.zeros((xyz.shape[0],), dtype=np.float64)  # reset errors conservatively
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
    ap.add_argument("--images_subdir", type=str, default="images_4", help="Which images dir to read")
    ap.add_argument("--out_name", type=str, default="points3D_dense.bin", help="Output filename under sparse/0/")
    ap.add_argument("--roma_model", type=str, default="outdoor", choices=["outdoor","indoor"], help="RoMa model variant")
    ap.add_argument("--cpu", action="store_true", help="Force CPU for RoMa (slow)")
    # EDGS knobs
    ap.add_argument("--num_refs", type=int, default=150, help="Reference frames (k-means over poses)")
    ap.add_argument("--nns_per_ref", type=int, default=3, help="Nearest neighbors per ref (use 3–5 for robustness)")
    ap.add_argument("--matches_per_ref", type=int, default=20000, help="Samples per ref after aggregation & threshold")
    ap.add_argument("--certainty_thresh", type=float, default=0.2, help="Min certainty to consider a pixel")
    ap.add_argument("--reproj_thresh", type=float, default=3.0, help="Max reprojection error (px)")
    ap.add_argument("--sampson_thresh", type=float, default=6.0, help="Max Sampson error (px^2) before triangulation (<=0 to disable)")
    ap.add_argument("--min_parallax_deg", type=float, default=0.1, help="Min parallax (deg); set 0 to disable")
    # Output shaping
    ap.add_argument("--voxel_size", type=float, default=0.0, help="Optional voxel size (scene units); keep small or 0")
    ap.add_argument("--max_points", type=int, default=0, help="Optional cap on total points (0 = unlimited)")
    ap.add_argument("--viz", action="store_true", help="Visualize input/output point clouds (Open3D)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    dense_init(args)
