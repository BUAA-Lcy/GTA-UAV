# -*- coding: utf-8 -*-

import os
import cv2
import time
import torch
import logging
import numpy as np
import torchvision.transforms.functional as F

from os.path import join
from .networks.lightglue.superpoint import SuperPoint
from .networks.lightglue.models.matchers.lightglue import LightGlue


class SparseSpLgMatcher:
    def __init__(self, device="cuda", logger=None):
        self.device = device
        self.logger = logger
        self.project_root = os.path.abspath(join(os.path.dirname(__file__), "..", ".."))

        sp_conf = {
            "max_num_keypoints_val": 2048,  # Increased from 512 to get more keypoints
            "legacy_sampling": False,
            "nms_radius": 4,
            "remove_borders": 4,
            "detection_threshold": 0.005,
        }
        lg_conf = {
            "filter_threshold": 0.0,
            "flash": False,
            "mp": False,
            "checkpointed": False,
        }
        self.sp = SuperPoint(sp_conf).eval().to(device)
        self.lg = LightGlue(lg_conf).eval().to(device)

        sp_loaded = self._try_load_weights(
            self.sp,
            [
                join(self.project_root, "pretrained", "gim", "superpoint_v1.pth"),
                join(self.project_root, "pretrained", "lightglue", "superpoint_v1.pth"),
                join(self.project_root, "weights", "superpoint_v1.pth"),
            ],
            "SuperPoint",
        )
        lg_loaded = self._try_load_weights(
            self.lg,
            [
                join(self.project_root, "pretrained", "gim", "superpoint_lightglue.pth"),
                join(self.project_root, "pretrained", "lightglue", "superpoint_lightglue.pth"),
                join(self.project_root, "weights", "superpoint_lightglue.pth"),
            ],
            "LightGlue",
        )
        if not (sp_loaded and lg_loaded):
            missing_parts = []
            if not sp_loaded:
                missing_parts.append("SuperPoint")
            if not lg_loaded:
                missing_parts.append("LightGlue")
            missing_str = "/".join(missing_parts) if missing_parts else "unknown"
            err_msg = (
                "sparse 模式初始化失败：缺少或无法加载权重("
                + missing_str
                + ")。请检查预训练文件是否存在且可读。"
            )
            self._log(logging.ERROR, err_msg)
            raise RuntimeError(err_msg)

        self.sparse_ransac_method = cv2.USAC_MAGSAC # 改为更先进的 USAC_MAGSAC
        self.sparse_ransac_reproj_threshold = 20.0 # 放宽重投影误差阈值，允许微小的像素偏移
        self.sparse_ransac_confidence = 0.99 # 提高置信度要求，让算法搜索更彻底
        self.sparse_ransac_max_iter = 1000 # 大幅增加 RANSAC 迭代次数，防止过早退出
        self.sparse_sp_max_edge = 1024  # Increased from 640 to extract features at higher res
        self.sparse_scales = (1.0, 0.75, 0.5, 1.25)  # Added more scales
        self.sparse_max_matches_per_scale = 1024  # Increased drastically to keep more matches
        self.sparse_max_total_matches = 4096      # Increased drastically
        self.sparse_min_inliers = 15 # 提高最低内点要求，如果二阶段抢救完还不到15个，就回退
        self.sparse_min_inlier_ratio = 0.001

        self.sparse_vis_dir = join(self.project_root, "Log", "sparse_bad_matches")
        self.sparse_vis_enable = True
        self.sparse_vis_max_save = 200
        self.sparse_vis_saved = 0
        os.makedirs(self.sparse_vis_dir, exist_ok=True)

        self.stats = {
            "n_queries": 0,
            "preproc_s": 0.0,
            "sp_detect_s": 0.0,
            "lg_match_s": 0.0,
            "ransac_f_s": 0.0,
            "ransac_h_s": 0.0,
            "total_s": 0.0,
            "n_samples": 0,
            "n_kept": 0,
            "empty_kpts": 0,
            "sparse_low_match": 0,
            "inliers_list": [],
        }

    def _log(self, level, msg, *args):
        if self.logger is not None:
            self.logger.log(level, msg, *args)

    def _try_load_weights(self, model, candidate_paths, model_name):
        for weight_path in candidate_paths:
            if not os.path.isfile(weight_path):
                continue
            try:
                state = torch.load(weight_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                if model_name == "LightGlue" and isinstance(state, dict):
                    fixed = {}
                    for k, v in state.items():
                        nk = k
                        if "self_attn." in nk:
                            nk = nk.replace("self_attn.", "transformers.")
                        if "cross_attn." in nk:
                            nk = nk.replace("cross_attn.", "transformers.")
                        fixed[nk] = v
                    state = fixed
                model.load_state_dict(state, strict=False)
                self._log(logging.INFO, "%s 权重加载成功: %s", model_name, weight_path)
                return True
            except Exception as e:
                self._log(logging.WARNING, "%s 权重加载失败: %s (%s)", model_name, weight_path, str(e))
        self._log(logging.WARNING, "%s 权重未找到，候选路径=%s", model_name, str(candidate_paths))
        return False

    def _save_sparse_bad_case(self, image0, image1, mk0, mk1, reason, inlier_mask=None, image1_vis=None, yaw_angle=None):
        if (not self.sparse_vis_enable) or (self.sparse_vis_saved >= self.sparse_vis_max_save):
            return
        try:
            img0 = image0[0].detach().cpu().permute(1, 2, 0).numpy()
            img1_src = image1_vis if image1_vis is not None else image1
            img1 = img1_src[0].detach().cpu().permute(1, 2, 0).numpy()
            img0 = np.clip(img0 * 255.0, 0, 255).astype(np.uint8)
            img1 = np.clip(img1 * 255.0, 0, 255).astype(np.uint8)
            if img0.ndim == 2 or img0.shape[2] == 1:
                img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
            if img1.ndim == 2 or img1.shape[2] == 1:
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)

            h0, w0 = img0.shape[:2]
            h1, w1 = img1.shape[:2]
            gap = 10
            canvas = np.zeros((max(h0, h1), w0 + gap + w1, 3), dtype=np.uint8)
            canvas[:h0, :w0] = img0
            canvas[:h1, w0 + gap:w0 + gap + w1] = img1

            n_matches = int(mk0.shape[0]) if isinstance(mk0, np.ndarray) else 0
            if n_matches > 0:
                max_draw = 120
                draw_idx = np.linspace(0, n_matches - 1, num=min(n_matches, max_draw), dtype=np.int32)
                inlier_bool = None
                if inlier_mask is not None:
                    inlier_bool = np.asarray(inlier_mask).reshape(-1).astype(bool)
                    if inlier_bool.shape[0] != n_matches:
                        inlier_bool = None
                for idx in draw_idx:
                    x0, y0 = int(round(float(mk0[idx, 0]))), int(round(float(mk0[idx, 1])))
                    x1, y1 = int(round(float(mk1[idx, 0]))), int(round(float(mk1[idx, 1])))
                    x1 += w0 + gap
                    c = (0, 255, 0) if (inlier_bool is None or inlier_bool[idx]) else (255, 80, 80)
                    cv2.line(canvas, (x0, y0), (x1, y1), c, 1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (x1, y1), 2, c, -1, lineType=cv2.LINE_AA)

            text = f"{reason} | matches={n_matches}"
            if inlier_mask is not None and n_matches > 0:
                text += f" | inliers={int(np.asarray(inlier_mask).reshape(-1).astype(bool).sum())}"
            if yaw_angle is not None:
                text += f" | yaw_rot={float(yaw_angle):.2f}"
            cv2.putText(canvas, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, lineType=cv2.LINE_AA)

            self.sparse_vis_saved += 1
            out_path = join(self.sparse_vis_dir, f"{self.sparse_vis_saved:05d}_{reason}.png")
            cv2.imwrite(out_path, canvas[..., ::-1])
        except Exception as e:
            self._log(logging.DEBUG, "保存 sparse 可视化失败: %s", str(e))

    def _yaw_to_angle(self, yaw0, yaw1):
        if yaw0 is None or yaw1 is None:
            return 0.0
        try:
            yaw0_f = float(yaw0)
            yaw1_f = float(yaw1)
        except (TypeError, ValueError):
            return 0.0
        if (not np.isfinite(yaw0_f)) or (not np.isfinite(yaw1_f)):
            return 0.0
        # Rotate image1 towards image0 orientation with a continuous angle.
        delta = yaw1_f - yaw0_f 
        return ((delta + 180.0) % 360.0) - 180.0

    def _rotate_image_with_angle(self, image, angle_deg):
        # image: [1, 1, H, W], rotate around center, keep same size.
        if abs(float(angle_deg)) < 1e-6:
            return image, None
        h, w = int(image.shape[-2]), int(image.shape[-1])
        center = ((w - 1.0) * 0.5, (h - 1.0) * 0.5)
        rot_m = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0).astype(np.float32)
        img_np = image[0, 0].detach().cpu().numpy()
        img_rot = cv2.warpAffine(
            img_np,
            rot_m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        img_rot_t = torch.from_numpy(img_rot).to(device=image.device, dtype=image.dtype)[None, None]
        inv_rot_m = cv2.invertAffineTransform(rot_m).astype(np.float32)
        return img_rot_t, inv_rot_m

    def _rotate_image_tensor_for_vis(self, image, angle_deg):
        # image: [1, C, H, W], keep size and black-pad out-of-bound area.
        if abs(float(angle_deg)) < 1e-6:
            return image
        img_np = image[0].detach().cpu().permute(1, 2, 0).numpy()
        h, w = img_np.shape[:2]
        center = ((w - 1.0) * 0.5, (h - 1.0) * 0.5)
        rot_m = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0).astype(np.float32)
        img_rot = cv2.warpAffine(
            img_np,
            rot_m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if img_rot.ndim == 2:
            img_rot = img_rot[..., None]
        return torch.from_numpy(img_rot).permute(2, 0, 1).contiguous().to(device=image.device, dtype=image.dtype)[None]

    def _warp_points_affine(self, pts, affine_mat):
        if pts.shape[0] == 0 or affine_mat is None:
            return pts.astype(np.float32)
        pts_h = np.concatenate([pts.astype(np.float32), np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        pts_out = (affine_mat @ pts_h.T).T
        return pts_out.astype(np.float32)

    def _run_matching_for_angles(self, image0, image1, yaw0, yaw1, candidate_angles, reproj_threshold, run_name="搜索", ransac_method=None):
        """提取公共逻辑：针对一组候选角度进行匹配，并返回最优结果和日志。"""
        if ransac_method is None:
            ransac_method = self.sparse_ransac_method
            
        best_H = np.eye(3)
        best_inliers = -1
        best_ratio = -1.0
        best_stats = None
        best_angle = 0.0
        rotation_search_logs = []

        img0_sp = image0 if image0.shape[1] == 1 else F.rgb_to_grayscale(image0)
        img1_sp = image1 if image1.shape[1] == 1 else F.rgb_to_grayscale(image1)

        h0_orig, w0_orig = img0_sp.shape[-2], img0_sp.shape[-1]
        h1_orig, w1_orig = img1_sp.shape[-2], img1_sp.shape[-1]
        max_edge = max(h0_orig, w0_orig, h1_orig, w1_orig)
        base_scale = float(self.sparse_sp_max_edge) / float(max_edge) if max_edge > self.sparse_sp_max_edge else 1.0
        scales = []
        for s in self.sparse_scales:
            s_eff = max(0.2, min(1.0, float(s) * base_scale))
            if not any(abs(s_eff - x) < 1e-3 for x in scales):
                scales.append(s_eff)

        for search_angle in candidate_angles:
            try:
                t_preproc_s = time.perf_counter()
                t_preproc = time.perf_counter() - t_preproc_s # Just to initialize
                t_sp_detect, t_lg_match, total_kpts0 = 0.0, 0.0, 0
                
                base_rot_angle = self._yaw_to_angle(yaw0, yaw1)
                rot_angle = ((base_rot_angle + search_angle + 180.0) % 360.0) - 180.0
                
                t_preproc_s = time.perf_counter()
                image1_vis_rot = self._rotate_image_tensor_for_vis(image1, rot_angle)
                t_preproc += time.perf_counter() - t_preproc_s

                mk0_buckets, mk1_buckets, ms_buckets = [], [], []

                for s_eff in scales:
                    t_preproc_s = time.perf_counter()
                    h0_sp = max(16, int(round(h0_orig * s_eff)))
                    w0_sp = max(16, int(round(w0_orig * s_eff)))
                    h1_base = max(16, int(round(h1_orig * s_eff)))
                    w1_base = max(16, int(round(w1_orig * s_eff)))
                    img0_s = img0_sp if (h0_sp == h0_orig and w0_sp == w0_orig) else F.resize(img0_sp, [h0_sp, w0_sp])
                    img1_base = img1_sp if (h1_base == h1_orig and w1_base == w1_orig) else F.resize(img1_sp, [h1_base, w1_base])
                    img1_s, inv_rot_m = self._rotate_image_with_angle(img1_base, rot_angle)
                    h1_sp, w1_sp = img1_s.shape[-2], img1_s.shape[-1]
                    t_preproc += time.perf_counter() - t_preproc_s

                    t_sp = time.perf_counter()
                    with torch.inference_mode():
                        pred0 = self.sp({"image": img0_s})
                        pred1 = self.sp({"image": img1_s})
                    t_sp_detect += time.perf_counter() - t_sp

                    kpts0_sp = pred0["keypoints"][0]
                    desc0_sp = pred0["descriptors"][0]
                    total_kpts0 += int(kpts0_sp.shape[0])
                    if kpts0_sp.shape[0] < 2: continue
                    
                    kpts1_sp = pred1["keypoints"][0]
                    desc1_sp = pred1["descriptors"][0]
                    if kpts1_sp.shape[0] < 2: continue

                    t_lg = time.perf_counter()
                    with torch.inference_mode():
                        lg_pred = self.lg(
                            {
                                "keypoints0": kpts0_sp[None],
                                "keypoints1": kpts1_sp[None],
                                "descriptors0": desc0_sp[None],
                                "descriptors1": desc1_sp[None],
                                "image_size0": torch.tensor([[w0_sp, h0_sp]], device=image0.device, dtype=kpts0_sp.dtype),
                                "image_size1": torch.tensor([[w1_sp, h1_sp]], device=image1.device, dtype=kpts1_sp.dtype),
                            }
                        )
                    t_lg_match += time.perf_counter() - t_lg

                    t_preproc_s = time.perf_counter()
                    if "matches" in lg_pred and len(lg_pred["matches"]) > 0:
                        matches = lg_pred["matches"][0]
                        mk0_i = kpts0_sp[matches[:, 0]].cpu().detach().numpy()
                        mk1_i = kpts1_sp[matches[:, 1]].cpu().detach().numpy()
                        ms_i = lg_pred["scores"][0].cpu().detach().numpy() if ("scores" in lg_pred and len(lg_pred["scores"]) > 0) else np.ones((mk0_i.shape[0],), dtype=np.float32)
                    else:
                        m0 = lg_pred["matches0"][0].cpu()
                        valid = m0 > -1
                        idx0 = torch.where(valid)[0]
                        idx1 = m0[valid]
                        mk0_i = kpts0_sp[idx0].cpu().detach().numpy()
                        mk1_i = kpts1_sp[idx1].cpu().detach().numpy()
                        ms_i = lg_pred["matching_scores0"][0].cpu().detach().numpy()[valid.numpy()] if "matching_scores0" in lg_pred else np.ones((mk0_i.shape[0],), dtype=np.float32)

                    if mk0_i.ndim == 1: mk0_i = mk0_i.reshape(-1, 2)
                    if mk1_i.ndim == 1: mk1_i = mk1_i.reshape(-1, 2)
                    mk0_i, mk1_i = mk0_i.astype(np.float32), mk1_i.astype(np.float32)
                    ms_i = ms_i.astype(np.float32) if isinstance(ms_i, np.ndarray) else np.ones((mk0_i.shape[0],), dtype=np.float32)
                    
                    if ms_i.shape[0] != mk0_i.shape[0] or mk0_i.shape[0] == 0:
                        t_preproc += time.perf_counter() - t_preproc_s
                        continue

                    mk1_i = self._warp_points_affine(mk1_i, inv_rot_m)
                    if w0_sp != w0_orig or h0_sp != h0_orig:
                        mk0_i[:, 0] *= float(w0_orig) / float(w0_sp)
                        mk0_i[:, 1] *= float(h0_orig) / float(h0_sp)
                    if w1_base != w1_orig or h1_base != h1_orig:
                        mk1_i[:, 0] *= float(w1_orig) / float(w1_base)
                        mk1_i[:, 1] *= float(h1_orig) / float(h1_base)

                    if mk0_i.shape[0] > self.sparse_max_matches_per_scale:
                        keep_idx = np.argsort(-ms_i)[: self.sparse_max_matches_per_scale]
                        mk0_i, mk1_i, ms_i = mk0_i[keep_idx], mk1_i[keep_idx], ms_i[keep_idx]

                    mk0_buckets.append(mk0_i)
                    mk1_buckets.append(mk1_i)
                    ms_buckets.append(ms_i)
                    t_preproc += time.perf_counter() - t_preproc_s

                t_preproc_s = time.perf_counter()
                if len(mk0_buckets) > 0:
                    mk0 = np.concatenate(mk0_buckets, axis=0)
                    mk1 = np.concatenate(mk1_buckets, axis=0)
                    ms = np.concatenate(ms_buckets, axis=0)
                    
                    if mk0.shape[0] > self.sparse_max_total_matches:
                        keep_idx = np.argsort(-ms)[: self.sparse_max_total_matches]
                        mk0, mk1 = mk0[keep_idx], mk1[keep_idx]
                else:
                    mk0 = np.empty((0, 2), dtype=np.float32)
                    mk1 = np.empty((0, 2), dtype=np.float32)
                t_preproc += time.perf_counter() - t_preproc_s

                H = np.eye(3)
                inliers, inlier_ratio, t_h, h_mask = 0, 0.0, 0.0, None
                
                if mk0.shape[0] >= 4 and mk1.shape[0] >= 4 and mk0.shape[1] == 2 and mk1.shape[1] == 2:
                    t_hs = time.perf_counter()
                    H_cand, h_mask = cv2.findHomography(
                        mk1, mk0, method=ransac_method,
                        ransacReprojThreshold=reproj_threshold, confidence=self.sparse_ransac_confidence, maxIters=self.sparse_ransac_max_iter,
                    )
                    t_h = time.perf_counter() - t_hs
                    if H_cand is not None:
                        H = H_cand
                        inliers = int(h_mask.sum()) if h_mask is not None else 0
                        inlier_ratio = float(inliers) / float(max(1, mk0.shape[0]))
                
                rotation_search_logs.append(f"[{run_name} {search_angle:>5.1f}°: Kpts0={total_kpts0:>4d} | Matches={mk0.shape[0]:>4d} | Inliers={inliers:>3d} | Ratio={inlier_ratio:.3f}]")

                if inliers > best_inliers or (inliers == best_inliers and inlier_ratio > best_ratio):
                    best_inliers, best_ratio, best_H, best_angle = inliers, inlier_ratio, H, rot_angle
                    best_stats = {
                        "preproc_s": t_preproc, "sp_detect_s": t_sp_detect, "lg_match_s": t_lg_match, "ransac_h_s": t_h,
                        "total_kpts0": total_kpts0, "n_kept": int(mk0.shape[0]), "mk0_shape": str(mk0.shape), "mk1_shape": str(mk1.shape),
                        "h_mask": h_mask, "image1_vis_rot": image1_vis_rot, "mk0": mk0, "mk1": mk1
                    }

            except Exception as e:
                self._log(logging.WARNING, "SP+LG %s 异常 (Angle: %.1f): %s", run_name, search_angle, str(e))
                rotation_search_logs.append(f"[{run_name} {search_angle:>5.1f}°: Exception]")
                continue

        return best_inliers, best_ratio, best_H, best_angle, best_stats, rotation_search_logs

    def match(self, image0, image1, yaw0=None, yaw1=None, rotate=True):
        t_total = time.perf_counter()
        
        # Phase 1: 4-way rotation with normal RANSAC threshold
        candidate_angles_p1 = [0.0, 90.0, 180.0, 270.0] if rotate else [0.0]
        thresh_p1 = self.sparse_ransac_reproj_threshold
        
        b_inliers, b_ratio, b_H, b_angle, b_stats, logs_p1 = self._run_matching_for_angles(
            image0, image1, yaw0, yaw1, candidate_angles_p1, thresh_p1, "一阶段"
        )
        all_logs = logs_p1
        
        # Phase 2: If the best match from Phase 1 has < 10 inliers and we are rotating, trigger 8-way fallback search
        if rotate and b_inliers < 10:
            self._log(logging.DEBUG, "一阶段最优内点数(%d) < 10，触发二阶段8向放宽搜索 (Threshold: 100.0, Method: RANSAC)", b_inliers)
            candidate_angles_p2 = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
            thresh_p2 = 100.0  # 大幅度放宽重投影误差
            
            # 使用传统的 cv2.RANSAC 代替 USAC_MAGSAC。
            # 原因是 USAC_MAGSAC 在点极少且分布差时可能会过度追求局部极小值（或者因为自由度问题失败），
            # 而朴素的 RANSAC 配合超大的 threshold，就是“强行”找能框住最多点的几何体，更容易在绝境中捞出点。
            method_p2 = cv2.RANSAC
            
            b_inliers_p2, b_ratio_p2, b_H_p2, b_angle_p2, b_stats_p2, logs_p2 = self._run_matching_for_angles(
                image0, image1, yaw0, yaw1, candidate_angles_p2, thresh_p2, "二阶段", ransac_method=method_p2
            )
            all_logs.extend(logs_p2)
            
            # 如果二阶段找到了更多的内点，就采用二阶段的结果
            if b_inliers_p2 > b_inliers:
                b_inliers, b_ratio, b_H, b_angle, b_stats = b_inliers_p2, b_ratio_p2, b_H_p2, b_angle_p2, b_stats_p2
                self._log(logging.DEBUG, "二阶段搜索成功拯救样本，内点数提升至 %d", b_inliers)
            else:
                self._log(logging.DEBUG, "二阶段搜索未能进一步提升内点数，保留一阶段结果")

        total_t = time.perf_counter() - t_total
        search_log_str = " | ".join(all_logs)

        if b_stats is None:
            self.stats["n_queries"] += 1
            self.stats["total_s"] += total_t
            return np.eye(3)
            
        H, inliers, inlier_ratio, rot_angle = b_H, b_inliers, b_ratio, b_angle
        
        if b_stats["n_kept"] < 4:
            self._save_sparse_bad_case(
                image0, image1, b_stats["mk0"], b_stats["mk1"], "low_match",
                image1_vis=b_stats["image1_vis_rot"], yaw_angle=rot_angle,
            )
            self.stats["sparse_low_match"] += 1
            if self.stats["sparse_low_match"] <= 5 or (self.stats["sparse_low_match"] % 200 == 0):
                self._log(logging.WARNING, "SP+LG 匹配点不足: mk0=%s mk1=%s，回退单位H", b_stats["mk0_shape"], b_stats["mk1_shape"])
            H = np.eye(3)
        elif inliers < self.sparse_min_inliers or inlier_ratio < self.sparse_min_inlier_ratio:
            self._save_sparse_bad_case(
                image0, image1, b_stats["mk0"], b_stats["mk1"], "low_quality",
                b_stats["h_mask"], image1_vis=b_stats["image1_vis_rot"], yaw_angle=rot_angle,
            )
            self._log(logging.DEBUG, "SP+LG H 质量不足，回退单位H: inliers=%d ratio=%.3f matches=%d", inliers, inlier_ratio, b_stats["n_kept"])
            H = np.eye(3)
            
        self.stats["n_queries"] += 1
        self.stats["preproc_s"] += b_stats["preproc_s"]
        self.stats["sp_detect_s"] += b_stats["sp_detect_s"]
        self.stats["lg_match_s"] += b_stats["lg_match_s"]
        self.stats["ransac_h_s"] += b_stats["ransac_h_s"]
        self.stats["total_s"] += total_t
        self.stats["n_samples"] += int(b_stats["total_kpts0"])
        self.stats["n_kept"] += b_stats["n_kept"]
        self.stats["inliers_list"].append(inliers)
        
        if search_log_str:
            self._log(logging.DEBUG, "Sparse两阶段搜索详情: %s => 最终选择Rot=%.1f°", search_log_str, float(rot_angle))
            
        self._log(
            logging.DEBUG,
            "with_match 子步骤(sparse-SP+LG): 预处理=%.6fs SP检测=%.6fs LG匹配=%.6fs RANSAC(F)=%.6fs 快速H=%.6fs 总计=%.6fs 关键点数=%d 匹配数=%d 内点数=%d yaw旋转角=%.2f mk0=%s mk1=%s",
            b_stats["preproc_s"], b_stats["sp_detect_s"], b_stats["lg_match_s"], 0.0, b_stats["ransac_h_s"], total_t, 
            int(b_stats["total_kpts0"]), b_stats["n_kept"], inliers, float(rot_angle), b_stats["mk0_shape"], b_stats["mk1_shape"],
        )
        return H

    def summarize_and_log(self):
        n = max(self.stats["n_queries"], 1)
        self._log(
            logging.INFO,
            "with_match 阶段平均耗时: 预处理=%.6fs SP检测=%.6fs LG匹配=%.6fs RANSAC(F)=%.6fs 快速H=%.6fs 总计=%.6fs",
            self.stats["preproc_s"] / n,
            self.stats["sp_detect_s"] / n,
            self.stats["lg_match_s"] / n,
            self.stats["ransac_f_s"] / n,
            self.stats["ransac_h_s"] / n,
            self.stats["total_s"] / n,
        )
        self._log(
            logging.INFO,
            "with_match 阶段样本统计: 平均关键点=%.1f 平均匹配保留=%.1f 空匹配=%d 低匹配回退=%d 总查询=%d",
            self.stats["n_samples"] / n,
            self.stats["n_kept"] / n,
            self.stats["empty_kpts"],
            self.stats["sparse_low_match"],
            self.stats["n_queries"],
        )
        
        # 统计并打印内点个数分布
        inliers_arr = np.array(self.stats["inliers_list"])
        if len(inliers_arr) > 0:
            total_valid = len(inliers_arr)
            gt_0 = np.sum(inliers_arr > 0)
            gt_10 = np.sum(inliers_arr > 10)
            gt_20 = np.sum(inliers_arr > 20)
            gt_50 = np.sum(inliers_arr > 50)
            
            self._log(
                logging.INFO,
                "Sparse内点数(Inliers)分布: >0点占比: %.2f%%, >10点占比: %.2f%%, >20点占比: %.2f%%, >50点占比: %.2f%%",
                (gt_0 / total_valid) * 100,
                (gt_10 / total_valid) * 100,
                (gt_20 / total_valid) * 100,
                (gt_50 / total_valid) * 100
            )
