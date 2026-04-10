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
    def __init__(
        self,
        device="cuda",
        logger=None,
        phase1_min_inliers=0,
        angle_score_inlier_offset=None,
        use_multi_scale=True,
        scales=None,
        multi_scale_mode="both",
        allow_upsample=False,
        cross_scale_dedup_radius=0.0,
        lightglue_profile="current",
        save_final_matches=False,
        save_final_matches_dir=None,
        save_final_matches_max=200,
    ):
        # Stable sparse defaults selected from the controlled VisLoc yaw-aligned tuning.
        if scales is None:
            scales = (1.0, 0.8, 0.6, 1.2)
        ransac_method = "RANSAC"
        ransac_reproj_threshold = 20.0
        ransac_confidence = 0.99
        ransac_max_iter = 1000
        sp_detection_threshold = 0.003
        sp_max_num_keypoints = 2048
        sp_nms_radius = 4
        sp_remove_borders = 4
        sp_max_edge = 1024
        max_matches_per_scale = 1024
        max_total_matches = 4096
        min_inliers = 15
        min_inlier_ratio = 0.001
        self.device = device
        self.logger = logger
        self.project_root = os.path.abspath(join(os.path.dirname(__file__), "..", ".."))

        ransac_zoo = {
            "RANSAC": cv2.RANSAC,
            "USAC_FAST": getattr(cv2, "USAC_FAST", cv2.RANSAC),
            "USAC_MAGSAC": getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
            "USAC_PROSAC": getattr(cv2, "USAC_PROSAC", cv2.RANSAC),
            "USAC_DEFAULT": getattr(cv2, "USAC_DEFAULT", cv2.RANSAC),
            "USAC_FM_8PTS": getattr(cv2, "USAC_FM_8PTS", cv2.RANSAC),
            "USAC_ACCURATE": getattr(cv2, "USAC_ACCURATE", cv2.RANSAC),
            "USAC_PARALLEL": getattr(cv2, "USAC_PARALLEL", cv2.RANSAC),
        }
        ransac_method_name = str(ransac_method).strip().upper()
        if ransac_method_name not in ransac_zoo:
            ransac_method_name = "RANSAC"

        sp_conf = {
            "max_num_keypoints_val": int(sp_max_num_keypoints),
            "legacy_sampling": False,
            "nms_radius": int(sp_nms_radius),
            "remove_borders": int(sp_remove_borders),
            "detection_threshold": float(sp_detection_threshold),
        }
        lightglue_profile_key = str(lightglue_profile).strip().lower()
        if lightglue_profile_key == "official_default":
            lg_conf = {
                "filter_threshold": 0.1,
                "flash": True,
                "mp": False,
                "depth_confidence": 0.95,
                "width_confidence": 0.99,
                "checkpointed": False,
            }
        else:
            lightglue_profile_key = "current"
            lg_conf = {
                "filter_threshold": 0.0,
                "flash": False,
                "mp": False,
                "depth_confidence": -1,
                "width_confidence": -1,
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

        self.sparse_ransac_method_name = ransac_method_name
        self.sparse_ransac_method = ransac_zoo[ransac_method_name]
        self.sparse_ransac_reproj_threshold = float(ransac_reproj_threshold)
        self.sparse_ransac_confidence = float(ransac_confidence)
        self.sparse_ransac_max_iter = int(ransac_max_iter)
        self.sparse_sp_max_edge = int(sp_max_edge)
        self.sparse_scales = self._normalize_scales(scales)
        self.sparse_max_matches_per_scale = int(max_matches_per_scale)
        self.sparse_max_total_matches = int(max_total_matches)
        self.sparse_phase1_min_inliers = max(0, int(phase1_min_inliers))
        self.sparse_angle_score_inlier_offset = None if angle_score_inlier_offset is None else float(angle_score_inlier_offset)
        self.sparse_use_multi_scale = bool(use_multi_scale)
        self.sparse_allow_upsample = bool(allow_upsample)
        multi_scale_mode_key = str(multi_scale_mode).strip().lower()
        if multi_scale_mode_key not in {"both", "query_only", "gallery_only"}:
            multi_scale_mode_key = "both"
        self.sparse_multi_scale_mode = multi_scale_mode_key
        self.sparse_cross_scale_dedup_radius = max(0.0, float(cross_scale_dedup_radius))
        self.sparse_lightglue_profile = lightglue_profile_key
        self.sparse_min_inliers = int(min_inliers)
        self.sparse_min_inlier_ratio = float(min_inlier_ratio)

        self.sparse_vis_dir = join(self.project_root, "Log", "sparse_bad_matches")
        self.sparse_vis_enable = True
        self.sparse_vis_max_save = 200
        self.sparse_vis_saved = 0
        os.makedirs(self.sparse_vis_dir, exist_ok=True)
        self.sparse_final_vis_dir = (
            str(save_final_matches_dir)
            if save_final_matches_dir
            else join(self.project_root, "Log", "visloc_sparse_final_matches")
        )
        self.sparse_final_vis_enable = bool(save_final_matches)
        self.sparse_final_vis_max_save = max(1, int(save_final_matches_max))
        self.sparse_final_vis_saved = 0
        if self.sparse_final_vis_enable:
            os.makedirs(self.sparse_final_vis_dir, exist_ok=True)

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
            "scale_summary": {},
        }
        self.last_match_info = None
        self.last_angle_results = []
        self.last_final_vis_path = None

    def _normalize_scales(self, scales):
        normalized = []
        seen = set()
        for value in scales:
            try:
                scale = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(scale) or scale <= 0.0:
                continue
            scale = round(scale, 6)
            if scale in seen:
                continue
            seen.add(scale)
            normalized.append(scale)
        if not normalized:
            normalized = [1.0]
        return tuple(normalized)

    def _format_scale_label(self, q_scale, g_scale):
        q_value = float(q_scale)
        g_value = float(g_scale)
        if abs(q_value - g_value) < 1e-6:
            return f"s{q_value:.2f}"
        return f"q{q_value:.2f}_g{g_value:.2f}"

    def _accumulate_scale_summary(self, scale_stats):
        if not scale_stats:
            return
        summary_dict = self.stats.setdefault("scale_summary", {})
        for item in scale_stats:
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            summary_entry = summary_dict.setdefault(
                label,
                {
                    "label": label,
                    "q_scale": float(item.get("q_scale", 0.0)),
                    "g_scale": float(item.get("g_scale", 0.0)),
                    "selected_queries": 0,
                    "matched_queries": 0,
                    "inlier_queries": 0,
                    "retained_matches_total": 0,
                    "inliers_total": 0,
                },
            )
            summary_entry["selected_queries"] += 1
            retained_matches = int(item.get("retained_matches", 0))
            inliers = int(item.get("inliers", 0))
            if retained_matches > 0:
                summary_entry["matched_queries"] += 1
            if inliers > 0:
                summary_entry["inlier_queries"] += 1
            summary_entry["retained_matches_total"] += retained_matches
            summary_entry["inliers_total"] += inliers

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

    def _save_sparse_case(
        self,
        image0,
        image1,
        mk0,
        mk1,
        reason,
        out_dir,
        counter_attr,
        max_save,
        enabled=True,
        inlier_mask=None,
        image1_vis=None,
        yaw_angle=None,
        case_name=None,
    ):
        if (not enabled) or (getattr(self, counter_attr) >= max_save):
            return None
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

            next_idx = getattr(self, counter_attr) + 1
            setattr(self, counter_attr, next_idx)
            case_slug = self._build_case_slug(case_name)
            filename_parts = [f"{next_idx:05d}"]
            if case_slug:
                filename_parts.append(case_slug)
            filename_parts.append(reason)
            out_path = join(out_dir, "_".join(filename_parts) + ".png")
            cv2.imwrite(out_path, canvas[..., ::-1])
            return out_path
        except Exception as e:
            self._log(logging.DEBUG, "保存 sparse 可视化失败: %s", str(e))
            return None

    def _build_case_slug(self, case_name):
        if case_name is None:
            return ""
        name = os.path.splitext(os.path.basename(str(case_name)))[0].strip()
        if not name:
            return ""
        sanitized = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in name)
        sanitized = "_".join(part for part in sanitized.split("_") if part)
        return sanitized[:120]

    def _compute_candidate_score(self, inliers, inlier_ratio):
        if self.sparse_angle_score_inlier_offset is None:
            return None
        return float(inlier_ratio) * max(float(inliers) - float(self.sparse_angle_score_inlier_offset), 0.0)

    def _prefer_candidate(self, inliers, inlier_ratio, best_inliers, best_ratio):
        current_score = self._compute_candidate_score(inliers, inlier_ratio)
        best_score = self._compute_candidate_score(best_inliers, best_ratio) if best_inliers >= 0 else None

        if current_score is not None and best_score is not None:
            if current_score > best_score:
                return True
            if current_score == best_score and inliers > best_inliers:
                return True
            if current_score == best_score and inliers == best_inliers and inlier_ratio > best_ratio:
                return True
            return False

        if inliers > best_inliers:
            return True
        if inliers == best_inliers and inlier_ratio > best_ratio:
            return True
        return False

    def _build_angle_result(self, phase, search_angle, rot_angle, matches, inliers, ratio, total_kpts0, H, status, score):
        H_value = None if H is None else np.asarray(H, dtype=np.float32).copy()
        return {
            "phase": int(phase),
            "search_angle": float(search_angle),
            "rot_angle": float(rot_angle),
            "matches": int(matches),
            "inliers": int(inliers),
            "ratio": float(ratio),
            "score": None if score is None else float(score),
            "total_kpts0": int(total_kpts0),
            "homography": H_value,
            "status": str(status),
            "selected": False,
        }

    def _save_sparse_bad_case(self, image0, image1, mk0, mk1, reason, inlier_mask=None, image1_vis=None, yaw_angle=None, case_name=None):
        return self._save_sparse_case(
            image0=image0,
            image1=image1,
            mk0=mk0,
            mk1=mk1,
            reason=reason,
            out_dir=self.sparse_vis_dir,
            counter_attr="sparse_vis_saved",
            max_save=self.sparse_vis_max_save,
            enabled=self.sparse_vis_enable,
            inlier_mask=inlier_mask,
            image1_vis=image1_vis,
            yaw_angle=yaw_angle,
            case_name=case_name,
        )

    def _save_sparse_final_case(self, image0, image1, mk0, mk1, reason, inlier_mask=None, image1_vis=None, yaw_angle=None, case_name=None):
        return self._save_sparse_case(
            image0=image0,
            image1=image1,
            mk0=mk0,
            mk1=mk1,
            reason=reason,
            out_dir=self.sparse_final_vis_dir,
            counter_attr="sparse_final_vis_saved",
            max_save=self.sparse_final_vis_max_save,
            enabled=self.sparse_final_vis_enable,
            inlier_mask=inlier_mask,
            image1_vis=image1_vis,
            yaw_angle=yaw_angle,
            case_name=case_name,
        )

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

    def _normalize_rotate_step(self, rotate, default_step=90.0):
        if isinstance(rotate, bool):
            return default_step if rotate else 0.0
        try:
            rotate_step = float(rotate)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(rotate_step):
            return default_step
        return max(0.0, abs(rotate_step))

    def _build_candidate_angles(self, rotate_step):
        rotate_step = self._normalize_rotate_step(rotate_step)
        if rotate_step <= 1e-6 or rotate_step >= 360.0:
            return [0.0]
        n_steps = int(np.floor((360.0 - 1e-6) / rotate_step)) + 1
        return [float(round(i * rotate_step, 6)) for i in range(n_steps)]

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

    def _build_scale_pairs(self, base_scale, max_scale):
        base_scale = max(0.2, float(base_scale))
        max_scale = max(base_scale, float(max_scale))
        if not self.sparse_use_multi_scale:
            return [(base_scale, base_scale)]

        scale_pairs = []
        for s in self.sparse_scales:
            s_eff = max(0.2, min(max_scale, float(s) * base_scale))
            if self.sparse_multi_scale_mode == "query_only":
                pair = (s_eff, base_scale)
            elif self.sparse_multi_scale_mode == "gallery_only":
                pair = (base_scale, s_eff)
            else:
                pair = (s_eff, s_eff)
            if not any(abs(pair[0] - x0) < 1e-3 and abs(pair[1] - x1) < 1e-3 for x0, x1 in scale_pairs):
                scale_pairs.append(pair)
        return scale_pairs if scale_pairs else [(base_scale, base_scale)]

    def _dedup_cross_scale_matches(self, mk0, mk1, ms, scale_ids=None):
        if (
            mk0.shape[0] <= 1
            or mk1.shape[0] != mk0.shape[0]
            or ms.shape[0] != mk0.shape[0]
            or self.sparse_cross_scale_dedup_radius <= 0.0
        ):
            if scale_ids is None:
                return mk0, mk1, ms
            return mk0, mk1, ms, scale_ids

        radius_sq = float(self.sparse_cross_scale_dedup_radius) ** 2
        order = np.argsort(-ms.astype(np.float32))
        suppressed = np.zeros((mk0.shape[0],), dtype=bool)
        keep_indices = []

        for idx in order:
            if suppressed[idx]:
                continue
            keep_indices.append(int(idx))
            dq = mk0[:, 0] - mk0[idx, 0]
            dq2 = dq * dq
            dy = mk0[:, 1] - mk0[idx, 1]
            qdist_sq = dq2 + dy * dy
            dg = mk1[:, 0] - mk1[idx, 0]
            dg2 = dg * dg
            gy = mk1[:, 1] - mk1[idx, 1]
            gdist_sq = dg2 + gy * gy
            dup_mask = (qdist_sq <= radius_sq) & (gdist_sq <= radius_sq)
            suppressed |= dup_mask
            suppressed[idx] = True

        keep_indices = np.asarray(keep_indices, dtype=np.int64)
        if scale_ids is None:
            return mk0[keep_indices], mk1[keep_indices], ms[keep_indices]
        return mk0[keep_indices], mk1[keep_indices], ms[keep_indices], scale_ids[keep_indices]

    def _run_matching_for_angles(self, image0, image1, yaw0, yaw1, candidate_angles, reproj_threshold, run_name="搜索", ransac_method=None):
        """提取公共逻辑：针对一组候选角度进行匹配，并返回最优结果和日志。"""
        if ransac_method is None:
            ransac_method = self.sparse_ransac_method
        phase_idx = 2 if "二阶段" in str(run_name) else 1
            
        best_H = np.eye(3)
        best_inliers = -1
        best_ratio = -1.0
        best_stats = None
        best_angle = 0.0
        rotation_search_logs = []
        angle_results = []

        img0_sp = image0 if image0.shape[1] == 1 else F.rgb_to_grayscale(image0)
        img1_sp = image1 if image1.shape[1] == 1 else F.rgb_to_grayscale(image1)

        h0_orig, w0_orig = img0_sp.shape[-2], img0_sp.shape[-1]
        h1_orig, w1_orig = img1_sp.shape[-2], img1_sp.shape[-1]
        max_edge = max(h0_orig, w0_orig, h1_orig, w1_orig)
        base_scale = float(self.sparse_sp_max_edge) / float(max_edge) if max_edge > self.sparse_sp_max_edge else 1.0
        max_scale = 1.0
        if self.sparse_allow_upsample and max_edge > 0:
            max_scale = max(1.0, float(self.sparse_sp_max_edge) / float(max_edge))
        scale_pairs = self._build_scale_pairs(base_scale, max_scale)
        scale_labels = [self._format_scale_label(q_scale, g_scale) for q_scale, g_scale in scale_pairs]

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

                mk0_buckets, mk1_buckets, ms_buckets, scale_id_buckets = [], [], [], []
                per_scale_stats = [
                    {
                        "label": scale_labels[scale_idx],
                        "q_scale": float(q_scale),
                        "g_scale": float(g_scale),
                        "retained_matches": 0,
                        "inliers": 0,
                    }
                    for scale_idx, (q_scale, g_scale) in enumerate(scale_pairs)
                ]

                for scale_idx, (q_scale, g_scale) in enumerate(scale_pairs):
                    t_preproc_s = time.perf_counter()
                    h0_sp = max(16, int(round(h0_orig * q_scale)))
                    w0_sp = max(16, int(round(w0_orig * q_scale)))
                    h1_base = max(16, int(round(h1_orig * g_scale)))
                    w1_base = max(16, int(round(w1_orig * g_scale)))
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

                    per_scale_stats[scale_idx]["retained_matches"] = int(mk0_i.shape[0])
                    mk0_buckets.append(mk0_i)
                    mk1_buckets.append(mk1_i)
                    ms_buckets.append(ms_i)
                    scale_id_buckets.append(np.full((mk0_i.shape[0],), scale_idx, dtype=np.int32))
                    t_preproc += time.perf_counter() - t_preproc_s

                t_preproc_s = time.perf_counter()
                if len(mk0_buckets) > 0:
                    mk0 = np.concatenate(mk0_buckets, axis=0)
                    mk1 = np.concatenate(mk1_buckets, axis=0)
                    ms = np.concatenate(ms_buckets, axis=0)
                    scale_ids = np.concatenate(scale_id_buckets, axis=0)
                    mk0, mk1, ms, scale_ids = self._dedup_cross_scale_matches(mk0, mk1, ms, scale_ids=scale_ids)
                    
                    if mk0.shape[0] > self.sparse_max_total_matches:
                        keep_idx = np.argsort(-ms)[: self.sparse_max_total_matches]
                        mk0, mk1, scale_ids = mk0[keep_idx], mk1[keep_idx], scale_ids[keep_idx]
                    if scale_ids.shape[0] == mk0.shape[0]:
                        for scale_idx in range(len(per_scale_stats)):
                            per_scale_stats[scale_idx]["retained_matches"] = int(np.sum(scale_ids == scale_idx))
                else:
                    mk0 = np.empty((0, 2), dtype=np.float32)
                    mk1 = np.empty((0, 2), dtype=np.float32)
                    scale_ids = np.empty((0,), dtype=np.int32)
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
                        if h_mask is not None and scale_ids.shape[0] == mk0.shape[0]:
                            h_mask_bool = np.asarray(h_mask).reshape(-1).astype(bool)
                            for scale_idx in range(len(per_scale_stats)):
                                scale_mask = scale_ids == scale_idx
                                if np.any(scale_mask):
                                    per_scale_stats[scale_idx]["inliers"] = int(np.sum(h_mask_bool[scale_mask]))
                
                rotation_search_logs.append(f"[{run_name} {search_angle:>5.1f}°: Kpts0={total_kpts0:>4d} | Matches={mk0.shape[0]:>4d} | Inliers={inliers:>3d} | Ratio={inlier_ratio:.3f}]")
                angle_results.append(
                    self._build_angle_result(
                        phase=phase_idx,
                        search_angle=search_angle,
                        rot_angle=rot_angle,
                        matches=mk0.shape[0],
                        inliers=inliers,
                        ratio=inlier_ratio,
                        score=self._compute_candidate_score(inliers, inlier_ratio),
                        total_kpts0=total_kpts0,
                        H=H,
                        status="ok",
                    )
                )

                if self._prefer_candidate(inliers, inlier_ratio, best_inliers, best_ratio):
                    best_inliers, best_ratio, best_H, best_angle = inliers, inlier_ratio, H, rot_angle
                    best_stats = {
                        "preproc_s": t_preproc, "sp_detect_s": t_sp_detect, "lg_match_s": t_lg_match, "ransac_h_s": t_h,
                        "total_kpts0": total_kpts0, "n_kept": int(mk0.shape[0]), "mk0_shape": str(mk0.shape), "mk1_shape": str(mk1.shape),
                        "h_mask": h_mask, "image1_vis_rot": image1_vis_rot, "mk0": mk0, "mk1": mk1,
                        "scale_stats": [
                            {
                                "label": str(item["label"]),
                                "q_scale": float(item["q_scale"]),
                                "g_scale": float(item["g_scale"]),
                                "retained_matches": int(item["retained_matches"]),
                                "inliers": int(item["inliers"]),
                            }
                            for item in per_scale_stats
                        ],
                    }

            except Exception as e:
                self._log(logging.WARNING, "SP+LG %s 异常 (Angle: %.1f): %s", run_name, search_angle, str(e))
                rotation_search_logs.append(f"[{run_name} {search_angle:>5.1f}°: Exception]")
                angle_results.append(
                    self._build_angle_result(
                        phase=phase_idx,
                        search_angle=search_angle,
                        rot_angle=rot_angle if "rot_angle" in locals() else 0.0,
                        matches=0,
                        inliers=0,
                        ratio=0.0,
                        score=None,
                        total_kpts0=0,
                        H=None,
                        status="exception",
                    )
                )
                continue

        return best_inliers, best_ratio, best_H, best_angle, best_stats, rotation_search_logs, angle_results

    def match(self, image0, image1, yaw0=None, yaw1=None, rotate=True, case_name=None, save_final_vis=None):
        t_total = time.perf_counter()
        self.last_final_vis_path = None
        rotate_step_p1 = self._normalize_rotate_step(rotate)
        if rotate_step_p1 <= 1e-6:
            yaw0 = None
            yaw1 = None
        selected_phase = 1
        
        # Phase 1: angle search with the configured rotation step.
        candidate_angles_p1 = self._build_candidate_angles(rotate_step_p1)
        thresh_p1 = self.sparse_ransac_reproj_threshold
        
        b_inliers, b_ratio, b_H, b_angle, b_stats, logs_p1, angle_results_p1 = self._run_matching_for_angles(
            image0, image1, yaw0, yaw1, candidate_angles_p1, thresh_p1, "一阶段"
        )
        all_logs = logs_p1
        all_angle_results = list(angle_results_p1)
        
        # Phase 2: if phase 1 is weak, retry with half the step and a looser RANSAC threshold.
        if rotate_step_p1 > 0 and b_inliers < self.sparse_phase1_min_inliers:
            rotate_step_p2 = rotate_step_p1 / 2.0
            self._log(
                logging.DEBUG,
                "一阶段最优内点数(%d) < %d，触发二阶段放宽搜索 (Step: %.1f°, Threshold: 100.0, Method: RANSAC)",
                b_inliers,
                self.sparse_phase1_min_inliers,
                rotate_step_p2,
            )
            candidate_angles_p2 = self._build_candidate_angles(rotate_step_p2)
            thresh_p2 = 100.0  # 大幅度放宽重投影误差
            
            # 使用传统的 cv2.RANSAC 代替 USAC_MAGSAC。
            # 原因是 USAC_MAGSAC 在点极少且分布差时可能会过度追求局部极小值（或者因为自由度问题失败），
            # 而朴素的 RANSAC 配合超大的 threshold，就是“强行”找能框住最多点的几何体，更容易在绝境中捞出点。
            method_p2 = cv2.RANSAC
            
            b_inliers_p2, b_ratio_p2, b_H_p2, b_angle_p2, b_stats_p2, logs_p2, angle_results_p2 = self._run_matching_for_angles(
                image0, image1, yaw0, yaw1, candidate_angles_p2, thresh_p2, "二阶段", ransac_method=method_p2
            )
            all_logs.extend(logs_p2)
            all_angle_results.extend(angle_results_p2)
            
            # 如果二阶段找到了更多的内点，就采用二阶段的结果
            if b_inliers_p2 > b_inliers:
                b_inliers, b_ratio, b_H, b_angle, b_stats = b_inliers_p2, b_ratio_p2, b_H_p2, b_angle_p2, b_stats_p2
                selected_phase = 2
                self._log(logging.DEBUG, "二阶段搜索成功拯救样本，内点数提升至 %d", b_inliers)
            else:
                self._log(logging.DEBUG, "二阶段搜索未能进一步提升内点数，保留一阶段结果")

        total_t = time.perf_counter() - t_total
        search_log_str = " | ".join(all_logs)
        self.last_angle_results = all_angle_results

        if b_stats is None:
            self.stats["n_queries"] += 1
            self.stats["total_s"] += total_t
            self.last_match_info = {
                "match_mode": "sparse",
                "phase": selected_phase,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "rot_angle": 0.0,
                "n_kept": 0,
                "identity_h_fallback": True,
                "fallback_to_center": True,
                "fallback_reason": "all_failed",
                "out_of_bounds": False,
                "projection_invalid": False,
                "final_vis_path": None,
                "scale_stats": [],
            }
            return np.eye(3)
            
        H, inliers, inlier_ratio, rot_angle = b_H, b_inliers, b_ratio, b_angle
        
        identity_h_fallback = False
        fallback_reason = None
        if b_stats["n_kept"] < 4:
            self._save_sparse_bad_case(
                image0, image1, b_stats["mk0"], b_stats["mk1"], "low_match",
                image1_vis=b_stats["image1_vis_rot"], yaw_angle=rot_angle, case_name=case_name,
            )
            self.stats["sparse_low_match"] += 1
            if self.stats["sparse_low_match"] <= 5 or (self.stats["sparse_low_match"] % 200 == 0):
                self._log(logging.WARNING, "SP+LG 匹配点不足: mk0=%s mk1=%s，回退单位H", b_stats["mk0_shape"], b_stats["mk1_shape"])
            H = np.eye(3)
            identity_h_fallback = True
            fallback_reason = "low_match"
        elif inliers < self.sparse_min_inliers or inlier_ratio < self.sparse_min_inlier_ratio:
            self._save_sparse_bad_case(
                image0, image1, b_stats["mk0"], b_stats["mk1"], "low_quality",
                b_stats["h_mask"], image1_vis=b_stats["image1_vis_rot"], yaw_angle=rot_angle, case_name=case_name,
            )
            self._log(logging.DEBUG, "SP+LG H 质量不足，回退单位H: inliers=%d ratio=%.3f matches=%d", inliers, inlier_ratio, b_stats["n_kept"])
            H = np.eye(3)
            identity_h_fallback = True
            fallback_reason = "low_quality"

        final_reason = f"final_phase{selected_phase}"
        if b_stats["n_kept"] < 4 or inliers < self.sparse_min_inliers or inlier_ratio < self.sparse_min_inlier_ratio:
            final_reason += "_fallback"
        save_final_vis_enabled = self.sparse_final_vis_enable if save_final_vis is None else bool(save_final_vis)
        if save_final_vis_enabled:
            self.last_final_vis_path = self._save_sparse_final_case(
                image0,
                image1,
                b_stats["mk0"],
                b_stats["mk1"],
                final_reason,
                b_stats["h_mask"],
                image1_vis=b_stats["image1_vis_rot"],
                yaw_angle=rot_angle,
                case_name=case_name,
            )
        else:
            self.last_final_vis_path = None
        for angle_result in self.last_angle_results:
            angle_result["selected"] = (
                int(angle_result.get("phase", 0)) == selected_phase
                and abs(float(angle_result.get("rot_angle", 0.0)) - float(rot_angle)) < 1e-6
                and int(angle_result.get("inliers", -1)) == int(inliers)
                and abs(float(angle_result.get("ratio", -1.0)) - float(inlier_ratio)) < 1e-6
            )
            
        self.stats["n_queries"] += 1
        self.stats["preproc_s"] += b_stats["preproc_s"]
        self.stats["sp_detect_s"] += b_stats["sp_detect_s"]
        self.stats["lg_match_s"] += b_stats["lg_match_s"]
        self.stats["ransac_h_s"] += b_stats["ransac_h_s"]
        self.stats["total_s"] += total_t
        self.stats["n_samples"] += int(b_stats["total_kpts0"])
        self.stats["n_kept"] += b_stats["n_kept"]
        self.stats["inliers_list"].append(inliers)
        self._accumulate_scale_summary(b_stats.get("scale_stats", []))
        
        if search_log_str:
            self._log(logging.DEBUG, "Sparse两阶段搜索详情: %s => 最终选择Rot=%.1f°", search_log_str, float(rot_angle))
        scale_stats = b_stats.get("scale_stats", [])
        if scale_stats:
            scale_log = " ; ".join(
                "{}(q={:.2f},g={:.2f}): kept={} inliers={} share={:.3f}".format(
                    str(item.get("label", "")),
                    float(item.get("q_scale", 0.0)),
                    float(item.get("g_scale", 0.0)),
                    int(item.get("retained_matches", 0)),
                    int(item.get("inliers", 0)),
                    float(item.get("inliers", 0)) / float(max(inliers, 1)),
                )
                for item in scale_stats
            )
            self._log(logging.DEBUG, "Sparse尺度贡献: %s", scale_log)
            
        self._log(
            logging.DEBUG,
            "with_match 子步骤(sparse-SP+LG): 预处理=%.6fs SP检测=%.6fs LG匹配=%.6fs RANSAC(F)=%.6fs 快速H=%.6fs 总计=%.6fs 关键点数=%d 匹配数=%d 内点数=%d yaw旋转角=%.2f mk0=%s mk1=%s",
            b_stats["preproc_s"], b_stats["sp_detect_s"], b_stats["lg_match_s"], 0.0, b_stats["ransac_h_s"], total_t, 
            int(b_stats["total_kpts0"]), b_stats["n_kept"], inliers, float(rot_angle), b_stats["mk0_shape"], b_stats["mk1_shape"],
        )
        self.last_match_info = {
            "match_mode": "sparse",
            "phase": selected_phase,
            "inliers": int(inliers),
            "inlier_ratio": float(inlier_ratio),
            "rot_angle": float(rot_angle),
            "n_kept": int(b_stats["n_kept"]),
            "identity_h_fallback": bool(identity_h_fallback),
            "fallback_to_center": bool(identity_h_fallback),
            "fallback_reason": fallback_reason,
            "out_of_bounds": False,
            "projection_invalid": False,
            "final_vis_path": self.last_final_vis_path,
            "scale_stats": [
                {
                    "label": str(item.get("label", "")),
                    "q_scale": float(item.get("q_scale", 0.0)),
                    "g_scale": float(item.get("g_scale", 0.0)),
                    "retained_matches": int(item.get("retained_matches", 0)),
                    "inliers": int(item.get("inliers", 0)),
                }
                for item in scale_stats
            ],
        }
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
        scale_summary = self.stats.get("scale_summary", {})
        if scale_summary:
            total_inliers = float(sum(int(item.get("inliers_total", 0)) for item in scale_summary.values()))
            for item in sorted(scale_summary.values(), key=lambda value: (-int(value.get("inliers_total", 0)), str(value.get("label", "")))):
                selected_queries = max(int(item.get("selected_queries", 0)), 1)
                inliers_total = int(item.get("inliers_total", 0))
                retained_total = int(item.get("retained_matches_total", 0))
                self._log(
                    logging.INFO,
                    "Sparse尺度贡献: label=%s q_scale=%.2f g_scale=%.2f selected_queries=%d matched_queries=%d inlier_queries=%d mean_retained_matches=%.2f mean_inliers=%.2f inlier_share=%.4f",
                    str(item.get("label", "")),
                    float(item.get("q_scale", 0.0)),
                    float(item.get("g_scale", 0.0)),
                    int(item.get("selected_queries", 0)),
                    int(item.get("matched_queries", 0)),
                    int(item.get("inlier_queries", 0)),
                    float(retained_total) / float(selected_queries),
                    float(inliers_total) / float(selected_queries),
                    float(inliers_total) / float(max(total_inliers, 1.0)),
                )

    def get_last_match_info(self):
        return self.last_match_info

    def get_last_angle_results(self):
        return self.last_angle_results
