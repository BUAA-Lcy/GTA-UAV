# -*- coding: utf-8 -*-
# @Author  : xuelun
# modified by jyx

import sys, os
import cv2
import torch
import argparse
import warnings
import logging
from .networks.lightglue.superpoint import SuperPoint
from .networks.lightglue.models.matchers.lightglue import LightGlue
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
        self.project_root = os.path.abspath(join(os.path.dirname(__file__), "..", ".."))
        self.sparse_weights_loaded = False
        if self.match_mode == "sparse":
            sp_conf = {
                "max_num_keypoints_val": 512,
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
            self.sparse_weights_loaded = bool(sp_loaded and lg_loaded)
            if not self.sparse_weights_loaded:
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
        self.sparse_num_samples = 256
        self.sparse_ransac_method = cv2.USAC_FAST
        self.sparse_ransac_reproj_threshold = 8.0
        self.sparse_ransac_confidence = 0.95
        self.sparse_ransac_max_iter = 50
        self.sparse_sp_max_edge = 640
        self.sparse_min_inliers = 8
        self.sparse_min_inlier_ratio = 0.20
        self.stats = {
            'mode': self.match_mode,
            'n_queries': 0,
            'preproc_s': 0.0,
            'dkm_match_s': 0.0,
            'dkm_sample_s': 0.0,
            'sp_detect_s': 0.0,
            'lg_match_s': 0.0,
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
            'sparse_low_match': 0,
        }
        self._log(logging.INFO, "with_match matcher mode: %s", self.match_mode)

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
                # 修正 LightGlue 旧版权重的键名以适配当前实现
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

    def match(self, image0, image1, vis=False):
        t_total = time.perf_counter()
        data = dict(color0=image0, color1=image1, image0=image0, image1=image1)

        if self.match_mode == "sparse":
            try:
                t_preproc_s = time.perf_counter()
                img0_sp = image0 if image0.shape[1] == 1 else F.rgb_to_grayscale(image0)
                img1_sp = image1 if image1.shape[1] == 1 else F.rgb_to_grayscale(image1)

                h0_orig, w0_orig = img0_sp.shape[-2], img0_sp.shape[-1]
                h1_orig, w1_orig = img1_sp.shape[-2], img1_sp.shape[-1]
                h0_sp, w0_sp = h0_orig, w0_orig
                h1_sp, w1_sp = h1_orig, w1_orig
                max_edge = max(h0_orig, w0_orig, h1_orig, w1_orig)
                if max_edge > self.sparse_sp_max_edge:
                    scale = float(self.sparse_sp_max_edge) / float(max_edge)
                    h0_sp = max(16, int(round(h0_orig * scale)))
                    w0_sp = max(16, int(round(w0_orig * scale)))
                    h1_sp = max(16, int(round(h1_orig * scale)))
                    w1_sp = max(16, int(round(w1_orig * scale)))
                    img0_sp = F.resize(img0_sp, [h0_sp, w0_sp])
                    img1_sp = F.resize(img1_sp, [h1_sp, w1_sp])
                t_preproc = time.perf_counter() - t_preproc_s

                t_sp = time.perf_counter()
                with torch.inference_mode():
                    pred0 = self.sp({"image": img0_sp})
                    pred1 = self.sp({"image": img1_sp})
                t_sp_detect = time.perf_counter() - t_sp

                kpts0_sp = pred0["keypoints"][0]
                kpts1_sp = pred1["keypoints"][0]
                desc0_sp = pred0["descriptors"][0]
                desc1_sp = pred1["descriptors"][0]

                if kpts0_sp.shape[0] < 2 or kpts1_sp.shape[0] < 2:
                    self._log(
                        logging.WARNING,
                        "SP 关键点不足: k0=%d k1=%d，回退单位H",
                        int(kpts0_sp.shape[0]),
                        int(kpts1_sp.shape[0]),
                    )
                    total_t = time.perf_counter() - t_total
                    self.stats['n_queries'] += 1
                    self.stats['preproc_s'] += t_preproc
                    self.stats['sp_detect_s'] += t_sp_detect
                    self.stats['total_s'] += total_t
                    self.stats['n_samples'] += int(kpts0_sp.shape[0])
                    return np.eye(3)

                t_lg = time.perf_counter()
                with torch.inference_mode():
                    lg_pred = self.lg({
                        "keypoints0": kpts0_sp[None],
                        "keypoints1": kpts1_sp[None],
                        "descriptors0": desc0_sp[None],
                        "descriptors1": desc1_sp[None],
                        "image_size0": torch.tensor([[w0_sp, h0_sp]], device=image0.device, dtype=kpts0_sp.dtype),
                        "image_size1": torch.tensor([[w1_sp, h1_sp]], device=image1.device, dtype=kpts1_sp.dtype),
                    })
                t_lg_match = time.perf_counter() - t_lg

                if "matches" in lg_pred and len(lg_pred["matches"]) > 0:
                    matches = lg_pred["matches"][0]
                    mk0 = kpts0_sp[matches[:, 0]].cpu().detach().numpy()
                    mk1 = kpts1_sp[matches[:, 1]].cpu().detach().numpy()
                else:
                    m0 = lg_pred["matches0"][0].cpu()
                    valid = m0 > -1
                    idx0 = torch.where(valid)[0]
                    idx1 = m0[valid]
                    mk0 = kpts0_sp[idx0].cpu().detach().numpy()
                    mk1 = kpts1_sp[idx1].cpu().detach().numpy()

                # 安全检查与类型转换，避免 OpenCV USAC 接口维度错误
                if mk0.ndim == 1:
                    mk0 = mk0.reshape(-1, 2)
                if mk1.ndim == 1:
                    mk1 = mk1.reshape(-1, 2)
                mk0 = mk0.astype(np.float32)
                mk1 = mk1.astype(np.float32)

                # 若SP输入发生缩放，需要将匹配点映射回原图坐标系后再做几何估计。
                if w0_sp != w0_orig or h0_sp != h0_orig:
                    mk0[:, 0] *= float(w0_orig) / float(w0_sp)
                    mk0[:, 1] *= float(h0_orig) / float(h0_sp)
                if w1_sp != w1_orig or h1_sp != h1_orig:
                    mk1[:, 0] *= float(w1_orig) / float(w1_sp)
                    mk1[:, 1] *= float(h1_orig) / float(h1_sp)

                if mk0.shape[0] < 4 or mk1.shape[0] < 4 or mk0.shape[1] != 2 or mk1.shape[1] != 2:
                    self.stats['sparse_low_match'] += 1
                    low_match_cnt = self.stats['sparse_low_match']
                    # 限频：前5次 + 每200次打印一次 warning，避免日志刷屏。
                    if low_match_cnt <= 5 or (low_match_cnt % 200 == 0):
                        self._log(
                            logging.WARNING,
                            "SP+LG 匹配点不足或维度异常: mk0=%s mk1=%s，回退单位H (累计=%d)",
                            str(mk0.shape),
                            str(mk1.shape),
                            low_match_cnt,
                        )
                    total_t = time.perf_counter() - t_total
                    self.stats['n_queries'] += 1
                    self.stats['preproc_s'] += t_preproc
                    self.stats['sp_detect_s'] += t_sp_detect
                    self.stats['lg_match_s'] += t_lg_match
                    self.stats['total_s'] += total_t
                    self.stats['n_samples'] += int(kpts0_sp.shape[0])
                    self.stats['n_kept'] += int(mk0.shape[0])
                    return np.eye(3)

                t_f = 0.0

                t_hs = time.perf_counter()
                H, h_mask = cv2.findHomography(
                    mk1, mk0,
                    method=self.sparse_ransac_method,
                    ransacReprojThreshold=self.sparse_ransac_reproj_threshold,
                    confidence=self.sparse_ransac_confidence,
                    maxIters=self.sparse_ransac_max_iter,
                )
                t_h = time.perf_counter() - t_hs
                if H is None:
                    H = np.eye(3)
                else:
                    inliers = int(h_mask.sum()) if h_mask is not None else 0
                    inlier_ratio = float(inliers) / float(max(1, mk0.shape[0]))
                    if inliers < self.sparse_min_inliers or inlier_ratio < self.sparse_min_inlier_ratio:
                        self._log(
                            logging.DEBUG,
                            "SP+LG H 质量不足，回退单位H: inliers=%d ratio=%.3f matches=%d",
                            inliers,
                            inlier_ratio,
                            int(mk0.shape[0]),
                        )
                        H = np.eye(3)

                total_t = time.perf_counter() - t_total
                self.stats['n_queries'] += 1
                self.stats['preproc_s'] += t_preproc
                self.stats['sp_detect_s'] += t_sp_detect
                self.stats['lg_match_s'] += t_lg_match
                self.stats['ransac_f_s'] += t_f
                self.stats['ransac_h_s'] += t_h
                self.stats['total_s'] += total_t
                self.stats['n_samples'] += int(kpts0_sp.shape[0])
                self.stats['n_kept'] += int(mk0.shape[0])
                self._log(
                    logging.DEBUG,
                    "with_match 子步骤(sparse-SP+LG): 预处理=%.6fs SP检测=%.6fs LG匹配=%.6fs RANSAC(F)=%.6fs 快速H=%.6fs 总计=%.6fs 关键点数=%d 匹配数=%d mk0=%s mk1=%s",
                    t_preproc, t_sp_detect, t_lg_match, t_f, t_h, total_t, int(kpts0_sp.shape[0]), int(mk0.shape[0]), str(mk0.shape), str(mk1.shape)
                )
                return H
            except Exception as e:
                self._log(logging.WARNING, "SP+LG 稀疏匹配异常: %s", str(e))
                total_t = time.perf_counter() - t_total
                self.stats['n_queries'] += 1
                self.stats['total_s'] += total_t
                return np.eye(3)

        b_ids, mconf, kpts0, kpts1 = None, None, None, None
        t0 = time.perf_counter()
        orig_width0, orig_height0, pad_left0, pad_right0, pad_top0, pad_bottom0 = get_padding_size(image0, 672, 896)
        orig_width1, orig_height1, pad_left1, pad_right1, pad_top1, pad_bottom1 = get_padding_size(image1, 672, 896)
        image0_ = torch.nn.functional.pad(image0, (pad_left0, pad_right0, pad_top0, pad_bottom0))
        image1_ = torch.nn.functional.pad(image1, (pad_left1, pad_right1, pad_top1, pad_bottom1))
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
        t_filter = time.perf_counter() - t4

        if len(kpts0) == 0 or len(kpts1) == 0:
            self.stats['empty_kpts'] += 1
            total_t = time.perf_counter() - t_total
            self.stats['n_queries'] += 1
            self.stats['preproc_s'] += t_preproc
            self.stats['dkm_match_s'] += 0.0
            self.stats['dkm_sample_s'] += 0.0
            self.stats['coord_s'] += t_coord
            self.stats['filter_s'] += t_filter
            self.stats['ransac_f_s'] += 0.0
            self.stats['ransac_h_s'] += 0.0
            self.stats['total_s'] += total_t
            self._log(logging.DEBUG,
                      "with_match 子步骤: 预处理=%.6fs 稠密匹配=%.6fs 采样=%.6fs 坐标换算=%.6fs 边界过滤=%.6fs RANSAC(F)=%.6fs RANSAC(H)=%.6fs 总计=%.6fs 样本数=%d 保留=%d",
                      t_preproc, 0.0, 0.0, t_coord, t_filter, 0.0, 0.0, total_t, int(sparse_matches.shape[0]) if 'sparse_matches' in locals() else 0, int(kpts0.shape[0]))
            return np.eye(3)

        if self.match_mode == "sparse":
            # 使用 SuperPoint + LightGlue 进行稀疏匹配，随后快速单应估计
            try:
                t_sp = time.perf_counter()
                pred0 = self.sp({"image": image0})
                pred1 = self.sp({"image": image1})
                t_sp_detect = time.perf_counter() - t_sp

                kpts0_sp = pred0["keypoints"]  # [B, M, 2]
                kpts1_sp = pred1["keypoints"]  # [B, N, 2]
                desc0_sp = pred0["descriptors"]  # [B, M, D]
                desc1_sp = pred1["descriptors"]  # [B, N, D]

                # 取 batch = 1
                kpts0_sp = kpts0_sp[0]
                kpts1_sp = kpts1_sp[0]
                desc0_sp = desc0_sp[0]
                desc1_sp = desc1_sp[0]

                h0, w0 = image0.shape[-2], image0.shape[-1]
                h1, w1 = image1.shape[-2], image1.shape[-1]
                t_lg = time.perf_counter()
                lg_pred = self.lg({
                    "keypoints0": kpts0_sp[None],
                    "keypoints1": kpts1_sp[None],
                    "descriptors0": desc0_sp[None],
                    "descriptors1": desc1_sp[None],
                    "image_size0": torch.tensor([[w0, h0]], device=image0.device, dtype=kpts0_sp.dtype),
                    "image_size1": torch.tensor([[w1, h1]], device=image1.device, dtype=kpts1_sp.dtype),
                })
                t_lg_match = time.perf_counter() - t_lg

                if "matches" in lg_pred and len(lg_pred["matches"]) > 0:
                    matches = lg_pred["matches"][0]  # [P, 2]
                    mk0 = kpts0_sp[matches[:, 0]].cpu().detach().numpy()
                    mk1 = kpts1_sp[matches[:, 1]].cpu().detach().numpy()
                else:
                    m0 = lg_pred["matches0"][0].cpu()
                    valid = m0 > -1
                    idx0 = torch.where(valid)[0]
                    idx1 = m0[valid]
                    mk0 = kpts0_sp[idx0].cpu().detach().numpy()
                    mk1 = kpts1_sp[idx1].cpu().detach().numpy()

                t_hs = time.perf_counter()
                H, _ = cv2.findHomography(
                    mk1,
                    mk0,
                    method=self.sparse_ransac_method,
                    ransacReprojThreshold=self.sparse_ransac_reproj_threshold,
                    confidence=self.sparse_ransac_confidence,
                    maxIters=self.sparse_ransac_max_iter,
                )
                t_h = time.perf_counter() - t_hs
                if H is None:
                    H = np.eye(3)

                total_t = time.perf_counter() - t_total
                self.stats['n_queries'] += 1
                self.stats['preproc_s'] += t_preproc
                self.stats['sp_detect_s'] += t_sp_detect
                self.stats['lg_match_s'] += t_lg_match
                self.stats['coord_s'] += 0.0
                self.stats['filter_s'] += 0.0
                self.stats['ransac_f_s'] += 0.0
                self.stats['ransac_h_s'] += t_h
                self.stats['total_s'] += total_t
                self.stats['n_samples'] += int(kpts0_sp.shape[0])
                self.stats['n_kept'] += int(mk0.shape[0])
                self._log(
                    logging.DEBUG,
                    "with_match 子步骤(sparse-SP+LG): 预处理=%.6fs SP检测=%.6fs LG匹配=%.6fs 快速H=%.6fs 总计=%.6fs 关键点数=%d 匹配数=%d",
                    t_preproc, t_sp_detect, t_lg_match, t_h, total_t, int(kpts0_sp.shape[0]), int(mk0.shape[0])
                )
                return H
            except Exception as e:
                # 发生异常时回退为单位矩阵，并记录失败
                self._log(logging.WARNING, "SP+LG 稀疏匹配异常: %s", str(e))
                total_t = time.perf_counter() - t_total
                self.stats['n_queries'] += 1
                self.stats['preproc_s'] += t_preproc
                self.stats['sp_detect_s'] += 0.0
                self.stats['lg_match_s'] += 0.0
                self.stats['ransac_h_s'] += 0.0
                self.stats['total_s'] += total_t
                return np.eye(3)

        try:
            t5 = time.perf_counter()
            _, mask = cv2.findFundamentalMat(kpts0.cpu().detach().numpy(),
                                            kpts1.cpu().detach().numpy(),
                                            cv2.USAC_MAGSAC, ransacReprojThreshold=1.0,
                                            confidence=0.999999, maxIters=10000)
            t_f = time.perf_counter() - t5
        except Exception as e:
            t_f = 0.0
            self.stats['f_fail'] += 1
            return np.eye(3)

        mask = mask.ravel() > 0

        data.update({
            'hw0_i': image0.shape[-2:],
            'hw1_i': image1.shape[-2:],
            'mkpts0_f': kpts0,
            'mkpts1_f': kpts1,
            'm_bids': b_ids,
            'mconf': mconf,
            'inliers': mask,
        })

        t6 = time.perf_counter()
        geom_info = compute_geom(data)
        t_h = time.perf_counter() - t6

        if vis:
            alpha = 0.5
            out = fast_make_matching_figure(data, b_id=0)
            overlay = fast_make_matching_overlay(data, b_id=0)
            out = cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0)
            cv2.imwrite(join('game4loc/matcher/assets/', f'match.png'), out[..., ::-1])
            wrapped_images = wrap_images(image0, image1, geom_info, "Homography")
            cv2.imwrite(join('game4loc/matcher/assets/', f'warp.png'), wrapped_images)

        total_t = time.perf_counter() - t_total
        self.stats['n_queries'] += 1
        self.stats['preproc_s'] += t_preproc
        self.stats['dkm_match_s'] += t_match
        self.stats['dkm_sample_s'] += t_sample
        self.stats['coord_s'] += t_coord
        self.stats['filter_s'] += t_filter
        self.stats['ransac_f_s'] += t_f
        self.stats['ransac_h_s'] += t_h
        self.stats['total_s'] += total_t
        self.stats['n_samples'] += int(sparse_matches.shape[0])
        self.stats['n_kept'] += int(kpts0.shape[0])
        self._log(logging.DEBUG,
                  "with_match 子步骤: 预处理=%.6fs 稠密匹配=%.6fs 采样=%.6fs 坐标换算=%.6fs 边界过滤=%.6fs RANSAC(F)=%.6fs RANSAC(H)=%.6fs 总计=%.6fs 样本数=%d 保留=%d",
                  t_preproc, t_match, t_sample, t_coord, t_filter, t_f, t_h, total_t, int(sparse_matches.shape[0]), int(kpts0.shape[0]))

        return np.array(geom_info["Homography"])
    
    def est_center(self, image0, image1, center_xy0, tl_xy0):
        t0 = time.perf_counter()
        image0 = image0.to(self.device)
        image1 = image1.to(self.device)
        if len(image0.shape) == 3:
            image0 = image0[None, ...]
        if len(image1.shape) == 3:
            image1 = image1[None, ...]

        image0 = image0 * 0.5 + 0.5
        image1 = image1 * 0.5 + 0.5

        H = self.match(image0, image1)

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
            return Xc_0, Yc_0

        proj_center_pixel = proj_pixel_homog[:2, 0] / denom
        x_pixel, y_pixel = float(proj_center_pixel[0]), float(proj_center_pixel[1])
        if (not np.isfinite(x_pixel)) or (not np.isfinite(y_pixel)) or x_pixel < -0.5 * w or x_pixel > 1.5 * w or y_pixel < -0.5 * h or y_pixel > 1.5 * h:
            self._log(logging.DEBUG, "with_match 投影越界(x=%.3f,y=%.3f,w=%d,h=%d)，回退原始中心点", x_pixel, y_pixel, w, h)
            return Xc_0, Yc_0

        X = Xtl_0 + x_pixel * s_x
        Y = Ytl_0 + y_pixel * s_y
        t_total = time.perf_counter() - t0
        self._log(logging.DEBUG, "with_match 位置估计耗时=%.6fs", t_total)

        return X, Y

    def summarize_and_log(self):
        n = max(self.stats['n_queries'], 1)
        avg_pre = self.stats['preproc_s'] / n
        avg_match = self.stats['dkm_match_s'] / n
        avg_sample = self.stats['dkm_sample_s'] / n
        avg_sp = self.stats['sp_detect_s'] / n
        avg_lg = self.stats['lg_match_s'] / n
        avg_coord = self.stats['coord_s'] / n
        avg_filter = self.stats['filter_s'] / n
        avg_f = self.stats['ransac_f_s'] / n
        avg_h = self.stats['ransac_h_s'] / n
        avg_total = self.stats['total_s'] / n
        avg_ns = self.stats['n_samples'] / n
        avg_keep = self.stats['n_kept'] / n
        self._log(logging.INFO, "with_match 模式: %s", self.match_mode)
        if self.match_mode == "sparse":
            self._log(logging.INFO,
                      "with_match 阶段平均耗时: 预处理=%.6fs SP检测=%.6fs LG匹配=%.6fs RANSAC(F)=%.6fs 快速H=%.6fs 总计=%.6fs",
                      avg_pre, avg_sp, avg_lg, avg_f, avg_h, avg_total)
            self._log(logging.INFO,
                      "with_match 阶段样本统计: 平均关键点=%.1f 平均匹配保留=%.1f 空匹配=%d 低匹配回退=%d 总查询=%d",
                      avg_ns, avg_keep, self.stats['empty_kpts'], self.stats['sparse_low_match'], self.stats['n_queries'])
        else:
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
