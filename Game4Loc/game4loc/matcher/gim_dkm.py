# -*- coding: utf-8 -*-
# @Author  : xuelun
# modified by jyx

import sys, os
import cv2
import torch
import argparse
import warnings
import logging
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from geopy.distance import geodesic
import time

from os.path import join
from .tools import get_padding_size
from .networks.loftr.loftr import LoFTR
from .networks.loftr.misc import lower_config
from .networks.loftr.config import get_cfg_defaults
from .networks.dkm.models.model_zoo.DKMv3 import DKMv3
from .sparse_sp_lg import SparseSpLgMatcher

DEFAULT_MIN_NUM_MATCHES = 4
DEFAULT_RANSAC_MAX_ITER = 10000
DEFAULT_RANSAC_CONFIDENCE = 0.999
DEFAULT_RANSAC_REPROJ_THRESHOLD = 8
DEFAULT_RANSAC_METHOD = "USAC_MAGSAC"

RANSAC_ZOO = {
    "RANSAC": cv2.RANSAC,
    "USAC_FAST": cv2.USAC_FAST,
    "USAC_MAGSAC": cv2.USAC_MAGSAC,
    "USAC_PROSAC": cv2.USAC_PROSAC,
    "USAC_DEFAULT": cv2.USAC_DEFAULT,
    "USAC_FM_8PTS": cv2.USAC_FM_8PTS,
    "USAC_ACCURATE": cv2.USAC_ACCURATE,
    "USAC_PARALLEL": cv2.USAC_PARALLEL,
}


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def resize_image(image, size, interp):
    assert interp.startswith('cv2_')
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    # elif interp.startswith('pil_'):
    #     interp = getattr(PIL.Image, interp[len('pil_'):].upper())
    #     resized = PIL.Image.fromarray(image.astype(np.uint8))
    #     resized = resized.resize(size, resample=interp)
    #     resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized


def fast_make_matching_figure(data, b_id):
    color0 = (data['color0'][b_id].permute(1, 2, 0).cpu().detach().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    color1 = (data['color1'][b_id].permute(1, 2, 0).cpu().detach().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    gray0 = cv2.cvtColor(color0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)
    kpts0 = data['mkpts0_f'].cpu().detach().numpy()
    kpts1 = data['mkpts1_f'].cpu().detach().numpy()
    mconf = data['mconf'].cpu().detach().numpy()
    inliers = data['inliers']

    rows = 2
    margin = 2
    (h0, w0), (h1, w1) = data['hw0_i'], data['hw1_i']
    h = max(h0, h1)
    H, W = margin * (rows + 1) + h * rows, margin * 3 + w0 + w1

    # canvas
    out = 255 * np.ones((H, W), np.uint8)

    wx = [margin, margin + w0, margin + w0 + margin, margin + w0 + margin + w1]
    hx = lambda row: margin * row + h * (row-1)
    out = np.stack([out] * 3, -1)

    sh = hx(row=1)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1

    sh = hx(row=2)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0[inliers], mkpts1[inliers]):
        c = (0, 255, 0)
        cv2.circle(out, (x0, y0 + sh), 3, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w0, y1 + sh), 3, c, -1, lineType=cv2.LINE_AA)

    return out


def fast_make_matching_overlay(data, b_id):
    color0 = (data['color0'][b_id].permute(1, 2, 0).cpu().detach().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    color1 = (data['color1'][b_id].permute(1, 2, 0).cpu().detach().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    gray0 = cv2.cvtColor(color0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)
    kpts0 = data['mkpts0_f'].cpu().detach().numpy()
    kpts1 = data['mkpts1_f'].cpu().detach().numpy()
    mconf = data['mconf'].cpu().detach().numpy()
    inliers = data['inliers']

    rows = 2
    margin = 2
    (h0, w0), (h1, w1) = data['hw0_i'], data['hw1_i']
    h = max(h0, h1)
    H, W = margin * (rows + 1) + h * rows, margin * 3 + w0 + w1

    # canvas
    out = 255 * np.ones((H, W), np.uint8)

    wx = [margin, margin + w0, margin + w0 + margin, margin + w0 + margin + w1]
    hx = lambda row: margin * row + h * (row-1)
    out = np.stack([out] * 3, -1)

    sh = hx(row=1)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1

    sh = hx(row=2)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0[inliers], mkpts1[inliers]):
        c = (0, 255, 0)
        cv2.line(out, (x0, y0 + sh), (x1 + margin + w0, y1 + sh), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x0, y0 + sh), 3, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w0, y1 + sh), 3, c, -1, lineType=cv2.LINE_AA)

    return out


def preprocess(image: np.ndarray, grayscale: bool = False, resize_max: int = None,
               dfactor: int = 8, resize_wh=(384, 384)):
    image = image.astype(np.float32, copy=False)
    size = image.shape[:2][::-1]
    scale = np.array([1.0, 1.0])

    if resize_max:
        scale = resize_max / max(size)
        if scale < 1.0:
            size_new = tuple(int(round(x*scale)) for x in size)
            image = resize_image(image, size_new, 'cv2_area')
            scale = np.array(size) / np.array(size_new)

    if grayscale:
        assert image.ndim == 2, image.shape
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = torch.from_numpy(image / 255.0).float()

    # assure that the size is divisible by dfactor
    # size_new = tuple(map(
    #         lambda x: int(x // dfactor * dfactor),
    #         image.shape[-2:]))
    # image = F.resize(image, size=size_new)
    image = F.resize(image, size=resize_wh)
    scale = 1 # np.array(size) / np.array(size_new)[::-1]
    return image, scale


def compute_geom(data,
                 ransac_method=DEFAULT_RANSAC_METHOD,
                 ransac_reproj_threshold=DEFAULT_RANSAC_REPROJ_THRESHOLD,
                 ransac_confidence=DEFAULT_RANSAC_CONFIDENCE,
                 ransac_max_iter=DEFAULT_RANSAC_MAX_ITER,
                 ) -> dict:

    mkpts0 = data["mkpts0_f"].cpu().detach().numpy()
    mkpts1 = data["mkpts1_f"].cpu().detach().numpy()

    if len(mkpts0) < 7 or len(mkpts1) < 7:
        return {"Homography": np.eye(3)}

    h1, w1 = data["hw0_i"]

    geo_info = {}

    try:
        F, inliers = cv2.findFundamentalMat(
            mkpts0,
            mkpts1,
            method=RANSAC_ZOO[ransac_method],
            ransacReprojThreshold=ransac_reproj_threshold,
            confidence=ransac_confidence,
            maxIters=ransac_max_iter,
        )
    except Exception as e:
        if np.isnan(mkpts0).any() or np.isnan(mkpts1).any() or \
            np.isinf(mkpts0).any() or np.isinf(mkpts1).any():
            print("Found NaN or Inf in points.")
        print('Error! Skip.')
        return {"Homography": np.eye(3)}

    if F is not None:
        geo_info["Fundamental"] = F.tolist()

    H, _ = cv2.findHomography(
        mkpts1,
        mkpts0,
        method=RANSAC_ZOO[ransac_method],
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=ransac_confidence,
        maxIters=ransac_max_iter,
    )
    if H is not None:
        geo_info["Homography"] = H.tolist()
        _, H1, H2 = cv2.stereoRectifyUncalibrated(
            mkpts0.reshape(-1, 2),
            mkpts1.reshape(-1, 2),
            F,
            imgSize=(w1, h1),
        )
        geo_info["H1"] = H1.tolist()
        geo_info["H2"] = H2.tolist()

    return geo_info


def wrap_images(img0, img1, geo_info, geom_type):
    img0 = img0[0].permute((1, 2, 0)).cpu().detach().numpy()[..., ::-1]
    img1 = img1[0].permute((1, 2, 0)).cpu().detach().numpy()[..., ::-1]

    h1, w1, _ = img0.shape
    h2, w2, _ = img1.shape

    rectified_image0 = img0
    rectified_image1 = None
    H = np.array(geo_info["Homography"])
    F = np.array(geo_info["Fundamental"])

    title = []
    if geom_type == "Homography":
        rectified_image1 = cv2.warpPerspective(
            img1, H, (img0.shape[1], img0.shape[0])
        )
        title = ["Image 0", "Image 1 - warped"]
    elif geom_type == "Fundamental":
        H1, H2 = np.array(geo_info["H1"]), np.array(geo_info["H2"])
        rectified_image0 = cv2.warpPerspective(img0, H1, (w1, h1))
        rectified_image1 = cv2.warpPerspective(img1, H2, (w2, h2))
        title = ["Image 0 - warped", "Image 1 - warped"]
    else:
        print("Error: Unknown geometry type")

    fig = plot_images(
        [rectified_image0.squeeze(), rectified_image1.squeeze()],
        title,
        dpi=300,
    )

    img = fig2im(fig)

    plt.close(fig)

    return img


def plot_images(imgs, titles=None, cmaps="gray", dpi=100, size=5, pad=0.5):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        dpi:
        size:
        pad:
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    figsize = (size * n, size * 6 / 5) if size is not None else None
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)

    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])

    fig.tight_layout(pad=pad)

    return fig


def fig2im(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf_ndarray = np.frombuffer(fig.canvas.buffer_rgba(), dtype="u1")
    # noinspection PyArgumentList
    im = buf_ndarray.reshape(h, w, 4)
    return im


class GimDKM:
    def __init__(
            self,
            device='cuda',
            logger=None,
            match_mode='sparse',
            sparse_angle_score_inlier_offset=None,
            sparse_use_multi_scale=True,
            sparse_scales=None,
            sparse_multi_scale_mode='both',
            sparse_allow_upsample=False,
            sparse_cross_scale_dedup_radius=0.0,
            sparse_lightglue_profile='current',
            sparse_save_final_vis=False,
            sparse_save_final_vis_dir=None,
            sparse_save_final_vis_max=200,
        ):
        self.match_mode = str(match_mode).lower()
        if self.match_mode not in {"dense", "sparse"}:
            self.match_mode = "sparse"

        ckpt = 'gim_dkm_100h.ckpt'
        self.model = DKMv3(weights=None, h=672, w=896)
        checkpoints_path = join('pretrained', 'gim', ckpt)

        # load state dict
        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
            if 'encoder.net.fc' in k:
                state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        self.model = self.model.eval().to(device)
        self.device = device
        self.logger = logger
        self.sparse_matcher = None
        if self.match_mode == "sparse":
            self.sparse_matcher = SparseSpLgMatcher(
                device=device,
                logger=logger,
                angle_score_inlier_offset=sparse_angle_score_inlier_offset,
                use_multi_scale=sparse_use_multi_scale,
                scales=sparse_scales,
                multi_scale_mode=sparse_multi_scale_mode,
                allow_upsample=sparse_allow_upsample,
                cross_scale_dedup_radius=sparse_cross_scale_dedup_radius,
                lightglue_profile=sparse_lightglue_profile,
                save_final_matches=sparse_save_final_vis,
                save_final_matches_dir=sparse_save_final_vis_dir,
                save_final_matches_max=sparse_save_final_vis_max,
            )
        self.stats = {
            'mode': self.match_mode,
            'n_queries': 0,
            'preproc_s': 0.0,
            'dkm_match_s': 0.0,
            'dkm_sample_s': 0.0,
            'coord_s': 0.0,
            'filter_s': 0.0,
            'ransac_f_s': 0.0,
            'ransac_h_s': 0.0,
            'total_s': 0.0,
            'n_samples': 0,
            'n_kept': 0,
            'f_fail': 0,
            'h_fail': 0,
            'empty_kpts': 0,
        }
        self.last_match_info = None
        self.last_angle_results = []
        self._log(logging.INFO, "with_match matcher mode: %s", self.match_mode)

    def _log(self, level, msg, *args):
        if self.logger is not None:
            self.logger.log(level, msg, *args)

    def _yaw_to_angle(self, yaw0, yaw1):
        """Convert yaw prior to rotation angle"""
        if yaw0 is None or yaw1 is None:
            return 0.0
        y0, y1 = float(yaw0), float(yaw1)
        # Normalize to [-180, 180)
        angle = ((y1 - y0 + 180.0) % 360.0) - 180.0
        return angle

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

    def _rotate_image_tensor(self, image_tensor, angle_deg):
        """Rotate a PyTorch image tensor by angle_deg and return the rotated tensor and its inverse affine matrix."""
        if abs(angle_deg) < 1e-3:
            return image_tensor, np.eye(2, 3, dtype=np.float32)
            
        h, w = image_tensor.shape[-2], image_tensor.shape[-1]
        
        center_x = w / 2.0
        center_y = h / 2.0
        
        rot_mat = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1.0)
        
        cos_val = np.abs(rot_mat[0, 0])
        sin_val = np.abs(rot_mat[0, 1])
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))
        
        rot_mat[0, 2] += (new_w / 2.0) - center_x
        rot_mat[1, 2] += (new_h / 2.0) - center_y
        
        inv_rot_mat = cv2.invertAffineTransform(rot_mat)
        
        # apply rotation
        device = image_tensor.device
        img_np = image_tensor[0].permute(1, 2, 0).cpu().numpy()
        
        rotated_np = cv2.warpAffine(img_np, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if len(rotated_np.shape) == 2:
            rotated_np = np.expand_dims(rotated_np, axis=2)
            
        rotated_tensor = torch.from_numpy(rotated_np).permute(2, 0, 1).unsqueeze(0).to(device)
        return rotated_tensor, inv_rot_mat

    def _warp_points_affine(self, pts, affine_mat):
        """Warp coordinates back using affine inverse matrix."""
        if pts.shape[0] == 0 or affine_mat is None:
            return pts.astype(np.float32)
        pts_h = np.concatenate([pts.astype(np.float32), np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        pts_out = (affine_mat @ pts_h.T).T
        return pts_out.astype(np.float32)

    def match(self, image0, image1, vis=False, yaw0=None, yaw1=None, rotate=True, case_name=None, save_final_vis=None):
        if self.match_mode == "sparse":
            H = self.sparse_matcher.match(
                image0,
                image1,
                yaw0=yaw0,
                yaw1=yaw1,
                rotate=rotate,
                case_name=case_name,
                save_final_vis=save_final_vis,
            )
            if self.sparse_matcher is not None and self.sparse_matcher.last_match_info is not None:
                self.last_match_info = dict(self.sparse_matcher.last_match_info)
            else:
                self.last_match_info = None
            if self.sparse_matcher is not None:
                self.last_angle_results = [dict(item) for item in self.sparse_matcher.get_last_angle_results()]
            else:
                self.last_angle_results = []
            return H

        t_total = time.perf_counter()
        self.last_angle_results = []
        rotate_step = self._normalize_rotate_step(rotate)
        if rotate_step <= 1e-6:
            yaw0 = None
            yaw1 = None
        
        # Dense mode angle search controlled by rotate step.
        candidate_angles = self._build_candidate_angles(rotate_step)
        
        best_H = np.eye(3)
        best_inliers = -1
        best_stats = None
        best_geom_info = None
        best_data = None
        best_rot_angle = 0.0
        
        rotation_search_logs = []
        
        base_rot_angle = self._yaw_to_angle(yaw0, yaw1)
        
        for search_angle in candidate_angles:
            try:
                rot_angle = ((base_rot_angle + search_angle + 180.0) % 360.0) - 180.0
                
                # Apply rotation to image1 (satellite image)
                image1_rot, inv_rot_m = self._rotate_image_tensor(image1, rot_angle)
                
                data = dict(color0=image0, color1=image1_rot, image0=image0, image1=image1_rot)
                b_ids, mconf, kpts0, kpts1 = None, None, None, None
                t0 = time.perf_counter()
                orig_width0, orig_height0, pad_left0, pad_right0, pad_top0, pad_bottom0 = get_padding_size(image0, 672, 896)
                orig_width1, orig_height1, pad_left1, pad_right1, pad_top1, pad_bottom1 = get_padding_size(image1_rot, 672, 896)
                image0_ = torch.nn.functional.pad(image0, (pad_left0, pad_right0, pad_top0, pad_bottom0))
                image1_ = torch.nn.functional.pad(image1_rot, (pad_left1, pad_right1, pad_top1, pad_bottom1))
                t_preproc = time.perf_counter() - t0

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t1 = time.perf_counter()
                    dense_matches, dense_certainty = self.model.match(image0_, image1_)
                    t_match = time.perf_counter() - t1
                    t2 = time.perf_counter()
                    num_samples = 5000
                    sparse_matches, mconf = self.model.sample(dense_matches, dense_certainty, num_samples)
                    t_sample = time.perf_counter() - t2

                height0, width0 = image0_.shape[-2:]
                height1, width1 = image1_.shape[-2:]

                t3 = time.perf_counter()
                kpts0 = sparse_matches[:, :2]
                kpts0 = torch.stack((
                    width0 * (kpts0[:, 0] + 1) / 2, height0 * (kpts0[:, 1] + 1) / 2), dim=-1,)
                kpts1 = sparse_matches[:, 2:]
                kpts1 = torch.stack((
                    width1 * (kpts1[:, 0] + 1) / 2, height1 * (kpts1[:, 1] + 1) / 2), dim=-1,)
                b_ids = torch.where(mconf[None])[0]
                t_coord = time.perf_counter() - t3

                t4 = time.perf_counter()
                kpts0 -= kpts0.new_tensor((pad_left0, pad_top0))[None]
                kpts1 -= kpts1.new_tensor((pad_left1, pad_top1))[None]
                mask_ = (kpts0[:, 0] > 0) & \
                       (kpts0[:, 1] > 0) & \
                       (kpts1[:, 0] > 0) & \
                       (kpts1[:, 1] > 0)
                mask_ = mask_ & \
                       (kpts0[:, 0] <= (orig_width0 - 1)) & \
                       (kpts1[:, 0] <= (orig_width1 - 1)) & \
                       (kpts0[:, 1] <= (orig_height0 - 1)) & \
                       (kpts1[:, 1] <= (orig_height1 - 1))

                mconf = mconf[mask_]
                b_ids = b_ids[mask_]
                kpts0 = kpts0[mask_]
                kpts1 = kpts1[mask_]
                
                # Inverse rotate the points back to original image1 coordinates
                if kpts1.shape[0] > 0:
                    kpts1_np = kpts1.cpu().detach().numpy()
                    kpts1_np = self._warp_points_affine(kpts1_np, inv_rot_m)
                    kpts1 = torch.from_numpy(kpts1_np).to(kpts1.device)
                
                t_filter = time.perf_counter() - t4

                if len(kpts0) == 0 or len(kpts1) == 0:
                    rotation_search_logs.append(f"[Angle {search_angle:>5.1f}°(Rot {rot_angle:>5.1f}°): Matches=0 | Inliers=0]")
                    continue

                try:
                    t5 = time.perf_counter()
                    _, mask = cv2.findFundamentalMat(kpts0.cpu().detach().numpy(),
                                                    kpts1.cpu().detach().numpy(),
                                                    cv2.USAC_MAGSAC, ransacReprojThreshold=1.0,
                                                    confidence=0.999999, maxIters=10000)
                    t_f = time.perf_counter() - t5
                except Exception as e:
                    rotation_search_logs.append(f"[Angle {search_angle:>5.1f}°(Rot {rot_angle:>5.1f}°): F_Fail]")
                    continue

                mask = mask.ravel() > 0
                inliers_count = int(mask.sum())

                data.update({
                    'hw0_i': image0.shape[-2:],
                    'hw1_i': image1.shape[-2:], # Use original shape for geom computation
                    'mkpts0_f': kpts0,
                    'mkpts1_f': kpts1,
                    'm_bids': b_ids,
                    'mconf': mconf,
                    'inliers': mask,
                })

                t6 = time.perf_counter()
                geom_info = compute_geom(data)
                t_h = time.perf_counter() - t6
                
                rotation_search_logs.append(f"[Angle {search_angle:>5.1f}°(Rot {rot_angle:>5.1f}°): Matches={kpts0.shape[0]:>4d} | Inliers={inliers_count:>4d}]")

                if inliers_count > best_inliers:
                    best_inliers = inliers_count
                    best_H = np.array(geom_info["Homography"])
                    best_geom_info = geom_info
                    best_data = data
                    best_rot_angle = rot_angle
                    best_stats = {
                        "preproc_s": t_preproc,
                        "dkm_match_s": t_match,
                        "dkm_sample_s": t_sample,
                        "coord_s": t_coord,
                        "filter_s": t_filter,
                        "ransac_f_s": t_f,
                        "ransac_h_s": t_h,
                        "n_samples": int(sparse_matches.shape[0]),
                        "n_kept": int(kpts0.shape[0]),
                    }

            except Exception as e:
                self._log(logging.WARNING, "DKM 稠密匹配异常 (Angle: %.1f): %s", search_angle, str(e))
                rotation_search_logs.append(f"[Angle {search_angle:>5.1f}°: Exception]")
                continue

        total_t = time.perf_counter() - t_total
        search_log_str = " | ".join(rotation_search_logs) if len(rotation_search_logs) > 1 else ""
        
        if best_stats is None:
            self.stats['empty_kpts'] += 1
            self.stats['n_queries'] += 1
            self.stats['total_s'] += total_t
            self.last_match_info = {
                "match_mode": "dense",
                "phase": 1,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "rot_angle": 0.0,
                "n_kept": 0,
                "identity_h_fallback": True,
                "fallback_to_center": True,
                "fallback_reason": "all_failed",
                "out_of_bounds": False,
                "projection_invalid": False,
            }
            if search_log_str:
                self._log(logging.DEBUG, "Dense(DKM)四向搜索详情: %s => 全部失败", search_log_str)
            return np.eye(3)
            
        if vis and best_data is not None:
            alpha = 0.5
            out = fast_make_matching_figure(best_data, b_id=0)
            overlay = fast_make_matching_overlay(best_data, b_id=0)
            out = cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0)
            cv2.imwrite(join('game4loc/matcher/assets/', f'match.png'), out[..., ::-1])
            wrapped_images = wrap_images(image0, image1, best_geom_info, "Homography")
            cv2.imwrite(join('game4loc/matcher/assets/', f'warp.png'), wrapped_images)

        self.stats['n_queries'] += 1
        self.stats['preproc_s'] += best_stats['preproc_s']
        self.stats['dkm_match_s'] += best_stats['dkm_match_s']
        self.stats['dkm_sample_s'] += best_stats['dkm_sample_s']
        self.stats['coord_s'] += best_stats['coord_s']
        self.stats['filter_s'] += best_stats['filter_s']
        self.stats['ransac_f_s'] += best_stats['ransac_f_s']
        self.stats['ransac_h_s'] += best_stats['ransac_h_s']
        self.stats['total_s'] += total_t # count the total 4-way search time
        self.stats['n_samples'] += best_stats['n_samples']
        self.stats['n_kept'] += best_stats['n_kept']
        
        if search_log_str:
            self._log(logging.DEBUG, "Dense(DKM)四向搜索详情: %s => 最终选择Rot=%.1f°", search_log_str, float(best_rot_angle))
            
        self._log(logging.DEBUG,
                  "with_match 子步骤(Dense-DKM): 预处理=%.6fs 稠密匹配=%.6fs 采样=%.6fs 坐标换算=%.6fs 边界过滤=%.6fs RANSAC(F)=%.6fs RANSAC(H)=%.6fs 总计=%.6fs 样本数=%d 保留=%d 内点数=%d",
                  best_stats['preproc_s'], best_stats['dkm_match_s'], best_stats['dkm_sample_s'], best_stats['coord_s'], best_stats['filter_s'], best_stats['ransac_f_s'], best_stats['ransac_h_s'], total_t, best_stats['n_samples'], best_stats['n_kept'], best_inliers)
        self.last_match_info = {
            "match_mode": "dense",
            "phase": 1,
            "inliers": int(best_inliers),
            "inlier_ratio": float(best_inliers) / float(max(int(best_stats["n_kept"]), 1)),
            "rot_angle": float(best_rot_angle),
            "n_kept": int(best_stats["n_kept"]),
            "identity_h_fallback": False,
            "fallback_to_center": False,
            "fallback_reason": None,
            "out_of_bounds": False,
            "projection_invalid": False,
        }
        self.last_angle_results = []

        return best_H

    def get_last_match_info(self):
        return self.last_match_info

    def get_last_angle_results(self):
        return self.last_angle_results
    
    def est_center(self, image0, image1, center_xy0, tl_xy0, yaw0=None, yaw1=None, rotate=True, case_name=None, save_final_vis=None):
        t0 = time.perf_counter()
        image0 = image0.to(self.device)
        image1 = image1.to(self.device)
        if len(image0.shape) == 3:
            image0 = image0[None, ...]
        if len(image1.shape) == 3:
            image1 = image1[None, ...]

        image0 = image0 * 0.5 + 0.5
        image1 = image1 * 0.5 + 0.5

        H = self.match(
            image0,
            image1,
            yaw0=yaw0,
            yaw1=yaw1,
            rotate=rotate,
            case_name=case_name,
            save_final_vis=save_final_vis,
        )
        if self.last_match_info is None:
            self.last_match_info = {
                "match_mode": self.match_mode,
                "phase": 1,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "rot_angle": 0.0,
                "n_kept": 0,
                "identity_h_fallback": False,
                "fallback_to_center": False,
                "fallback_reason": None,
                "out_of_bounds": False,
                "projection_invalid": False,
            }
        else:
            self.last_match_info = dict(self.last_match_info)
            self.last_match_info.setdefault("inlier_ratio", 0.0)
            self.last_match_info.setdefault("identity_h_fallback", False)
            self.last_match_info.setdefault("fallback_to_center", False)
            self.last_match_info.setdefault("fallback_reason", None)
            self.last_match_info.setdefault("out_of_bounds", False)
            self.last_match_info.setdefault("projection_invalid", False)

        h, w = image0.shape[2:]

        Xtl_0, Ytl_0 = tl_xy0
        Xc_0, Yc_0 = center_xy0

        s_x = (Xc_0 - Xtl_0) / (w / 2)
        s_y = (Yc_0 - Ytl_0) / (h / 2)

        center_pixel = np.array([w / 2, h / 2, 1]).reshape(3, 1)

        proj_pixel_homog = np.dot(H, center_pixel)
        denom = proj_pixel_homog[2, 0]
        if not np.isfinite(denom) or abs(float(denom)) < 1e-6:
            self._log(logging.DEBUG, "with_match 投影分母异常，回退原始中心点")
            self.last_match_info["projection_invalid"] = True
            self.last_match_info["fallback_to_center"] = True
            self.last_match_info["fallback_reason"] = "projection_invalid"
            return Xc_0, Yc_0

        proj_center_pixel = proj_pixel_homog[:2, 0] / denom
        x_pixel, y_pixel = float(proj_center_pixel[0]), float(proj_center_pixel[1])
        if (not np.isfinite(x_pixel)) or (not np.isfinite(y_pixel)) or x_pixel < -0.5 * w or x_pixel > 1.5 * w or y_pixel < -0.5 * h or y_pixel > 1.5 * h:
            self._log(logging.DEBUG, "with_match 投影越界(x=%.3f,y=%.3f,w=%d,h=%d)，回退原始中心点", x_pixel, y_pixel, w, h)
            self.last_match_info["out_of_bounds"] = True
            self.last_match_info["fallback_to_center"] = True
            self.last_match_info["fallback_reason"] = "out_of_bounds"
            return Xc_0, Yc_0

        X = Xtl_0 + x_pixel * s_x
        Y = Ytl_0 + y_pixel * s_y
        t_total = time.perf_counter() - t0
        self._log(logging.DEBUG, "with_match 位置估计耗时=%.6fs", t_total)

        return X, Y

    def summarize_and_log(self):
        if self.match_mode == "sparse" and self.sparse_matcher is not None:
            self._log(logging.INFO, "with_match 模式: %s", self.match_mode)
            self.sparse_matcher.summarize_and_log()
            return

        n = max(self.stats['n_queries'], 1)
        avg_pre = self.stats['preproc_s'] / n
        avg_match = self.stats['dkm_match_s'] / n
        avg_sample = self.stats['dkm_sample_s'] / n
        avg_coord = self.stats['coord_s'] / n
        avg_filter = self.stats['filter_s'] / n
        avg_f = self.stats['ransac_f_s'] / n
        avg_h = self.stats['ransac_h_s'] / n
        avg_total = self.stats['total_s'] / n
        avg_ns = self.stats['n_samples'] / n
        avg_keep = self.stats['n_kept'] / n
        self._log(logging.INFO, "with_match 模式: %s", self.match_mode)
        self._log(logging.INFO,
                  "with_match 阶段平均耗时: 预处理=%.6fs 稠密匹配=%.6fs 采样=%.6fs 坐标换算=%.6fs 边界过滤=%.6fs RANSAC(F)=%.6fs RANSAC(H)=%.6fs 总计=%.6fs",
                  avg_pre, avg_match, avg_sample, avg_coord, avg_filter, avg_f, avg_h, avg_total)
        self._log(logging.INFO,
                  "with_match 阶段样本统计: 平均采样数=%.1f 平均保留数=%.1f 失败(F)=%d 失败(H)=%d 空匹配=%d 总查询=%d",
                  avg_ns, avg_keep, self.stats['f_fail'], self.stats['h_fail'], self.stats['empty_kpts'], self.stats['n_queries'])



if __name__ == '__main__':
    device='cuda'

    matcher = GimDKM(device)

    # name0 = 'visloc_0427_sate'
    # name1 = 'visloc_0427_drone'
    # drone_lon_lat = 119.9267128, 32.22234999
    # sate_center_lon_lat = 119.92657586206897, 32.222450393667984
    # sate_tl_lon_lat = 119.9252028045977, 32.22382368089981

    name0 = 'gta_10645_sate'
    name1 = 'gta_10645_drone'
    drone_xy = 3712.3463380211374, 1911.5782488067728
    sate_center_xy = 3628.8, 1900.8
    sate_tl_xy = 3456.0, 1728.0

    postfix = '.png'
    image_dir = join('game4loc/matcher/assets')
    img_path0 = join(image_dir, name0 + postfix)
    img_path1 = join(image_dir, name1 + postfix)

    image0 = read_image(img_path0)
    image1 = read_image(img_path1)
    image0, scale0 = preprocess(image0)
    image1, scale1 = preprocess(image1)

    image0 = image0.to(device)[None]
    image1 = image1.to(device)[None]

    matcher.match(image0, image1, vis=True)

    # est_lon_lat = matcher.est_center(image0, image1, sate_center_lon_lat, sate_tl_lon_lat)
    # drone_lat_lon = drone_lon_lat[1], drone_lon_lat[0]
    # sate_center_lat_lon = sate_center_lon_lat[1], sate_center_lon_lat[0]
    # est_lat_lon = est_lon_lat[1], est_lon_lat[0]
    # print(f'Before error: {geodesic(drone_lat_lon, sate_center_lat_lon).meters}, After error: {geodesic(drone_lat_lon, est_lat_lon).meters}')

    def cal_dis(xy0, xy1):
        x0, y0 = xy0
        x1, y1 = xy1
        return ((x1-x0)**2 + (y1-y0)**2)**0.5

    est_xy = matcher.est_center(image0, image1, sate_center_xy, sate_tl_xy)
    print(f'Before error: {cal_dis(drone_xy, sate_center_xy)}, After error: {cal_dis(drone_xy, est_xy)}')
