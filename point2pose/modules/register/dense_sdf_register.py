"""Dense-primary registration: the visible depth surface against the object's TSDF.

The cluster-RANSAC register fits the pose to the sparse tracked points and polishes it against
the TSDF afterwards. Here the dense fit decides: every depth pixel inside the mask, subsampled,
is laid onto the TSDF by a Gauss-Newton on the signed distance, seeded by the sparse answer when
there is one and by the previous pose always, and the lower-cost seed wins if enough of the
surface agrees. When the sparse step has nothing (a hand over the tracked corners), the previous
pose alone seeds the fit, which is the hold under occlusion. The dense agreement — residuals in
metres per surface point and an inlier mask — rides on the stats under ``dense_*`` keys, so the
pipeline's lost test, jump guard and map-growth gate can judge a frame by its whole surface.

The TSDF query and the normal equations run in torch on the device: the volume is normalised
(value = signed distance / truncation margin), so residuals are scaled back to metres.
"""

from __future__ import annotations

import numpy as np
import torch

from point2pose.core.module_registry import REGISTER
from point2pose.modules.register.svd_cluster_ransac_register import (
    SVDClusterRANSACRegister,
    _left_update_SE3,
)
from point2pose.utils.camera import extract_cropped_point_cloud
from point2pose.utils.transform import inverse_SE3


@REGISTER.register_module("dense_sdf")
class DenseSDFRegister(SVDClusterRANSACRegister):
    dense_hold = True  # the front end calls register() even without sparse correspondences

    def __init__(self, config=None):
        super().__init__(config)
        config = config or {}
        self._dense_iters = int(config.get("dense_iters", 20))
        self._dense_max_pts = int(config.get("dense_max_pts", 2000))
        self._dense_min_pts = int(config.get("dense_min_pts", 60))
        self._dense_step_trans_clip = float(config.get("dense_step_trans_clip", 0.05))
        self._dense_step_rot_clip = float(config.get("dense_step_rot_clip", 0.5))
        self._dense_damping = float(config.get("dense_damping", 1e-6))
        self._dense_huber_m = float(config.get("dense_huber_m", 0.01))
        self._dense_tau_m = float(config.get("dense_inlier_tau_m", 0.008))
        self._dense_min_support = float(config.get("dense_min_support", 0.3))
        self._dense_min_inlier_ratio = float(config.get("dense_min_inlier_ratio", 0.5))
        self._dense_veto_ratio = float(config.get("dense_veto_ratio", 1.5))
        self._dense_veto_floor_m = float(config.get("dense_veto_floor_m", 0.005))
        self._dense_mad_scale = float(config.get("dense_mad_scale", 2.5))
        self._dense_device = str(config.get("dense_device", "cuda" if torch.cuda.is_available() else "cpu"))
        self._vol_cache: dict = {}

    # ----------------------------------------------------------------- the register entry point
    def register(
        self,
        src_pcd,
        tgt_pcd,
        init_pose=None,
        sigma_src=None,
        sigma_tgt=None,
        sigma=None,
        prev_T=None,
        prev_frame=None,
        cur_frame=None,
        obj=None,
        obj_id=0,
        mode="f2m",
    ):
        src_pcd = np.asarray(src_pcd, dtype=np.float32).reshape(-1, 3)
        tgt_pcd = np.asarray(tgt_pcd, dtype=np.float32).reshape(-1, 3)
        if src_pcd.shape[0] >= 3:
            T_sparse, stats = super().register(
                src_pcd,
                tgt_pcd,
                init_pose=init_pose,
                sigma_src=sigma_src,
                sigma_tgt=sigma_tgt,
                sigma=sigma,
                prev_T=prev_T,
                prev_frame=prev_frame,
                cur_frame=cur_frame,
                obj=obj,
                obj_id=obj_id,
                mode=mode,
            )
        else:
            T_sparse = np.asarray(init_pose if init_pose is not None else (prev_T if prev_T is not None else np.eye(4)))
            n = src_pcd.shape[0]
            stats = {
                "clusters": [],
                "best_cluster_idx": -1,
                "remaining_mask": np.ones(n, dtype=bool),
                "inliers": np.zeros(n, dtype=bool),
                "residuals": np.ones(n) * -1.0,
            }
        dense = {"applied": False, "reason": ""}
        stats["dense"] = dense
        vol = self._volume(obj)
        if vol is None or cur_frame is None:
            dense["reason"] = "no_sdf" if vol is None else "no_frame"
            return T_sparse, stats
        pts = self._dense_points(cur_frame, obj_id)
        if pts is None or pts.shape[0] < self._dense_min_pts:
            dense["reason"] = "too_few_points"
            return T_sparse, stats
        sparse_ok = int(stats.get("best_cluster_idx", -1)) >= 0 and int(
            np.count_nonzero(stats.get("inliers", np.zeros(0)))
        ) >= int(self._min_inliers)
        seeds = []
        if sparse_ok:
            seeds.append(("sparse", np.asarray(T_sparse, dtype=np.float64)))
        if prev_T is not None:
            seeds.append(("previous", np.asarray(prev_T, dtype=np.float64)))
        if not seeds:
            dense["reason"] = "no_seed"
            return T_sparse, stats
        pts_t = torch.as_tensor(pts, dtype=torch.float32, device=self._dense_device)
        best = None
        for name, T_seed in seeds:
            T_c2o, info = self._dense_refine(pts_t, inverse_SE3(T_seed), vol)
            if T_c2o is None:
                continue
            info["seed"] = name
            if best is None or info["cost"] < best[1]["cost"]:
                best = (T_c2o, info)
        if best is None:
            dense["reason"] = "no_convergence"
            return T_sparse, stats
        T_c2o, info = best
        dense.update({k: v for k, v in info.items() if k not in ("residuals", "inliers")})
        if info["support"] < self._dense_min_support or info["inlier_ratio"] < self._dense_min_inlier_ratio:
            dense["reason"] = "no_agreement"
            return T_sparse, stats
        # The tracked points keep their identity from frame to frame; the depth surface under the
        # mask does not (a finger on the object is depth under the mask). A dense answer that moves
        # the sparse inliers off their observations by more than the sparse answer did is refused.
        if sparse_ok:
            inl = np.asarray(stats["inliers"], dtype=bool)
            T_dense = inverse_SE3(T_c2o)
            r_dense = np.linalg.norm(
                src_pcd[inl] @ T_dense[:3, :3].T + T_dense[:3, 3] - tgt_pcd[inl], axis=1
            )
            r_sparse = np.linalg.norm(
                src_pcd[inl] @ np.asarray(T_sparse)[:3, :3].T + np.asarray(T_sparse)[:3, 3] - tgt_pcd[inl], axis=1
            )
            dense["sparse_median_m"] = float(np.median(r_dense))
            dense["sparse_median_before_m"] = float(np.median(r_sparse))
            if np.median(r_dense) > max(self._dense_veto_ratio * np.median(r_sparse), self._dense_veto_floor_m):
                dense["reason"] = "vetoed_by_tracks"
                return T_sparse, stats
        dense["applied"] = True
        stats["dense_residuals"] = info["residuals"]
        stats["dense_inliers"] = info["inliers"]
        return inverse_SE3(T_c2o), stats

    # ----------------------------------------------------------------- the volume on the device
    def _volume(self, obj):
        sdf = getattr(obj, "sdf", None) if obj is not None else None
        if not sdf or "tsdf" not in sdf:
            return None
        stamp = int(getattr(obj, "sdf_num_integrated", 0))
        cached = self._vol_cache.get(id(obj))
        if cached is not None and cached["stamp"] == stamp:
            return cached
        tsdf = np.asarray(sdf["tsdf"], dtype=np.float32)
        voxel = float(sdf["voxel_size"])
        volume = getattr(obj, "sdf_volume", None)
        trunc = float(getattr(volume, "_trunc_margin", 5.0 * voxel))
        cached = {
            "stamp": stamp,
            "vol": torch.as_tensor(tsdf, device=self._dense_device),
            "origin": torch.as_tensor(np.asarray(sdf["vol_origin"], dtype=np.float32), device=self._dense_device),
            "voxel": voxel,
            "trunc": trunc,
            "dims": torch.as_tensor(np.array(tsdf.shape, dtype=np.int64), device=self._dense_device),
        }
        self._vol_cache[id(obj)] = cached
        return cached

    def _dense_points(self, cur_frame, obj_id):
        try:
            pts = extract_cropped_point_cloud(
                cur_frame,
                obj_id,
                min_depth=self._select_3d_dist_min_depth,
                max_depth=self._select_3d_dist_max_depth,
                fill_missing_depth=self._select_3d_dist_fill_missing_depth,
                window_size=self._select_3d_dist_window_size,
                min_neighbors=self._select_3d_dist_min_neighbors,
            )
        except Exception:
            return None
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] > self._dense_max_pts:
            pts = pts[:: int(np.ceil(pts.shape[0] / float(self._dense_max_pts)))]
        return pts

    # ----------------------------------------------------------------- signed distance, in metres
    def _sdf(self, vol, pts_obj):
        """Trilinear signed distance (metres) and its exact gradient at ``pts_obj`` (n, 3), plus validity."""
        g = (pts_obj - vol["origin"]) / vol["voxel"]
        i0 = torch.floor(g).long()
        f = g - i0.to(g.dtype)
        valid = ((i0 >= 0) & (i0 + 1 < vol["dims"])).all(dim=1)
        i0 = torch.minimum(torch.clamp(i0, min=0), vol["dims"] - 2)
        x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
        v = vol["vol"]
        c000 = v[x0, y0, z0]
        c100 = v[x0 + 1, y0, z0]
        c010 = v[x0, y0 + 1, z0]
        c110 = v[x0 + 1, y0 + 1, z0]
        c001 = v[x0, y0, z0 + 1]
        c101 = v[x0 + 1, y0, z0 + 1]
        c011 = v[x0, y0 + 1, z0 + 1]
        c111 = v[x0 + 1, y0 + 1, z0 + 1]
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        c00 = c000 * (1 - fx) + c100 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        value = c0 * (1 - fz) + c1 * fz
        dx = ((c100 - c000) * (1 - fy) + (c110 - c010) * fy) * (1 - fz) + ((c101 - c001) * (1 - fy) + (c111 - c011) * fy) * fz
        dy = (c10 - c00) * (1 - fz) + (c11 - c01) * fz
        dz = c1 - c0
        scale = vol["trunc"]  # normalised units -> metres
        grad = torch.stack([dx, dy, dz], dim=1) * (scale / vol["voxel"])
        # Inside the truncation band the surface is where the value crosses zero; a point sitting
        # at the band's edge (|value| = 1) says nothing about the surface and carries no gradient.
        valid = valid & (value.abs() < 0.999) & (grad.norm(dim=1) > 1e-6)
        return value * scale, grad, valid

    def _dense_refine(self, pts_cur, T_cur2obj_init, vol):
        """Gauss-Newton on the signed distance of the visible surface, left-multiplicative updates on
        the camera-to-object transform; Huber weights, clipped steps, accept only descent."""
        dev = pts_cur.device
        pts_cur = pts_cur.float()
        T = torch.as_tensor(np.asarray(T_cur2obj_init, dtype=np.float64), dtype=torch.float32, device=dev)
        n_all = pts_cur.shape[0]
        # The segmenter module enters a bfloat16 autocast for the whole process; a 6x6 normal
        # equation on millimetre residuals is not a place for it.
        autocast_off = torch.autocast(device_type=dev.type, enabled=False)
        autocast_off.__enter__()
        try:
            return self._dense_refine_fp32(pts_cur, T, vol, n_all, dev)
        finally:
            autocast_off.__exit__(None, None, None)

    def _dense_refine_fp32(self, pts_cur, T, vol, n_all, dev):

        def evaluate(T_):
            pts_obj = pts_cur @ T_[:3, :3].T + T_[:3, 3]
            r, g, valid = self._sdf(vol, pts_obj)
            if int(valid.sum()) < self._dense_min_pts:
                return None
            r_v, g_v, x_v = r[valid], g[valid], pts_obj[valid]
            r_abs = r_v.abs()
            med = r_abs.median()
            mad = (r_abs - med).abs().median()
            keep = r_abs <= med + self._dense_mad_scale * mad + 1e-4  # the trimmed set, as the parent trims
            w = torch.ones_like(r_v)
            big = r_abs > self._dense_huber_m
            w[big] = self._dense_huber_m / r_abs[big]
            w = w * keep.to(w.dtype)
            if int(keep.sum()) < self._dense_min_pts:
                return None
            cost = float((w * r_v * r_v).sum() / keep.sum())
            return r_v, g_v, x_v, w, valid, cost

        state = evaluate(T)
        if state is None:
            return None, {"reason": "no_support"}
        best_T, best_cost = T.clone(), state[5]
        for _ in range(self._dense_iters):
            r_v, g_v, x_v, w, valid, cost = state
            # d r / d xi for a left update Exp(xi) T, xi = (translation, rotation): [g, x × g]
            J = torch.cat([g_v, torch.cross(x_v, g_v, dim=1)], dim=1)
            Jw = J * w.sqrt()[:, None]
            rw = r_v * w.sqrt()
            H = Jw.T @ Jw + self._dense_damping * torch.eye(6, device=dev)
            b = Jw.T @ rw
            try:
                dxi = -torch.linalg.solve(H, b)
            except RuntimeError:
                break
            if not torch.isfinite(dxi).all():
                break
            dxi[:3] = dxi[:3].clamp(-self._dense_step_trans_clip, self._dense_step_trans_clip)
            dxi[3:] = dxi[3:].clamp(-self._dense_step_rot_clip, self._dense_step_rot_clip)
            stepped = None
            for scale in (1.0, 0.5, 0.25):
                T_try = torch.as_tensor(
                    _left_update_SE3(T.double().cpu().numpy(), (scale * dxi).double().cpu().numpy()),
                    dtype=torch.float32,
                    device=dev,
                )
                cand = evaluate(T_try)
                if cand is not None and cand[5] < cost:
                    stepped = (T_try, cand)
                    break
            if stepped is None:
                break
            T, state = stepped
            if state[5] < best_cost:
                best_T, best_cost = T.clone(), state[5]
            if float(dxi[:3].norm()) < 1e-4 and float(dxi[3:].norm()) < 1e-3:
                break
        final = evaluate(best_T)
        if final is None:
            return None, {"reason": "no_support"}
        r_v, _, _, _, valid, cost = final
        residuals = np.zeros(n_all, dtype=np.float64)
        inliers = np.zeros(n_all, dtype=bool)
        valid_np = valid.cpu().numpy()
        residuals[valid_np] = r_v.abs().cpu().numpy()
        inliers[valid_np] = (r_v.abs() <= self._dense_tau_m).cpu().numpy()
        support = float(valid_np.mean())
        info = {
            "cost": float(cost),
            "support": support,
            "inlier_ratio": float(inliers[valid_np].mean()) if valid_np.any() else 0.0,
            "mean_abs_m": float(residuals[inliers].mean()) if inliers.any() else float("inf"),
            "n_pts": int(n_all),
            "residuals": residuals,
            "inliers": inliers,
        }
        return best_T.double().cpu().numpy(), info
