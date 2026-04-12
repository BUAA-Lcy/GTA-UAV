import json
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import gc
import time
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from sklearn.metrics import average_precision_score
from geopy.distance import geodesic

from ..matcher.gim_dkm import GimDKM
from ..orientation import load_vop_checkpoint, normalize_angle_deg, select_angle_result_with_vop
from ..verification import (
    build_confidence_candidate_record,
    load_confidence_verifier,
    select_candidate_with_confidence,
)


def annotate_final_match_visualization(image_path, distance_m, logger=None):
    if not image_path:
        return
    try:
        distance_value = float(distance_m)
    except (TypeError, ValueError):
        return
    if not np.isfinite(distance_value) or (not os.path.isfile(image_path)):
        return

    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            return
        text = f"Distance={distance_value:.2f}m"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        pad = 10
        rect_top = max(0, image.shape[0] - (text_h + baseline + pad * 2))
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (0, rect_top),
            (min(image.shape[1] - 1, text_w + pad * 2), image.shape[0] - 1),
            (0, 0, 0),
            thickness=-1,
        )
        image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0.0)
        cv2.putText(
            image,
            text,
            (pad, image.shape[0] - baseline - pad),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.imwrite(image_path, image)
    except Exception as exc:
        if logger is not None:
            logger.debug("为最终匹配图写入 Distance 失败: %s", str(exc))


def sdm(query_loc, sdmk_list, index, gallery_loc_xy_list, s=0.001):
    query_lat, query_lon = query_loc

    sdm_list = []

    for k in sdmk_list:
        sdm_nom = 0.0
        sdm_den = 0.0
        for i in range(k):
            idx = index[i]
            gallery_lat, gallery_lon = gallery_loc_xy_list[idx]
            d = geodesic((query_lat, query_lon), (gallery_lat, gallery_lon)).meters
            sdm_nom += (k - i) / np.exp(s * d)
            sdm_den += (k - i)
        sdm_list.append(sdm_nom/sdm_den)
    return sdm_list


def get_dis(query_loc, index, gallery_loc_xy_list, disk_list, match_loc=None):
    query_lat, query_lon = query_loc
    dis_list = []
    for k in disk_list:
        dis_sum = 0.0
        for i in range(k):
            idx = index[i]
            gallery_lat, gallery_lon = gallery_loc_xy_list[idx]
            dis = geodesic((query_lat, query_lon), (gallery_lat, gallery_lon)).meters
            dis_sum += dis

        # For matcher estimated location
        if k == 1 and match_loc != None:
            match_lat, match_lon = match_loc
            dis_match = geodesic((query_lat, query_lon), (match_lat, match_lon)).meters
            dis_list.append(dis_match)
        else:
            dis_list.append(dis_sum / k)

    return dis_list


def get_dis_target(query_loc, target_loc):
    query_lat, query_lon = query_loc
    target_lat, target_lon = target_loc
    return geodesic((query_lat, query_lon), (query_lat, target_lon)).meters


def project_match_center_from_h(H, image0, center_xy0, tl_xy0):
    if H is None:
        return None

    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        return None

    h, w = image0.shape[-2:]
    Xtl_0, Ytl_0 = tl_xy0
    Xc_0, Yc_0 = center_xy0

    s_x = (Xc_0 - Xtl_0) / (w / 2)
    s_y = (Yc_0 - Ytl_0) / (h / 2)

    center_pixel = np.array([w / 2, h / 2, 1.0], dtype=np.float64).reshape(3, 1)
    proj_pixel_homog = np.dot(H, center_pixel)
    denom = proj_pixel_homog[2, 0]
    if not np.isfinite(denom) or abs(float(denom)) < 1e-6:
        return None

    proj_center_pixel = proj_pixel_homog[:2, 0] / denom
    x_pixel, y_pixel = float(proj_center_pixel[0]), float(proj_center_pixel[1])
    if (
        (not np.isfinite(x_pixel))
        or (not np.isfinite(y_pixel))
        or x_pixel < -0.5 * w
        or x_pixel > 1.5 * w
        or y_pixel < -0.5 * h
        or y_pixel > 1.5 * h
    ):
        return None

    X = Xtl_0 + x_pixel * s_x
    Y = Ytl_0 + y_pixel * s_y
    return X, Y


def get_top10(index, gallery_list):
    top10 = []
    for i in range(10):
        idx = index[i]
        top10.append(gallery_list[idx])
    return top10


def _build_unique_candidate_angles(angle_results):
    candidate_angles = []
    seen = set()
    for angle_result in angle_results:
        if str(angle_result.get("status", "")) != "ok":
            continue
        if angle_result.get("homography") is None:
            continue
        angle = round(normalize_angle_deg(float(angle_result.get("rot_angle", 0.0))), 6)
        if angle in seen:
            continue
        seen.add(angle)
        candidate_angles.append(float(angle))
    return candidate_angles


def _rotate_query_tensor(query_tensor, angle_deg):
    if abs(float(angle_deg)) < 1e-6:
        return query_tensor
    fill_value = -1.0
    if query_tensor.ndim == 3:
        return TF.rotate(
            query_tensor,
            angle=float(angle_deg),
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=[fill_value] * int(query_tensor.shape[0]),
        )
    if query_tensor.ndim == 4:
        rotated = [
            TF.rotate(
                sample,
                angle=float(angle_deg),
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=[fill_value] * int(sample.shape[0]),
            )
            for sample in query_tensor
        ]
        return torch.stack(rotated, dim=0)
    raise ValueError(f"Unsupported query tensor shape for rotation: {tuple(query_tensor.shape)}")


def _prefer_match_candidate(inliers, inlier_ratio, best_inliers, best_ratio):
    if inliers > best_inliers:
        return True
    if inliers == best_inliers and inlier_ratio > best_ratio:
        return True
    return False


def _to_jsonable_candidate_record(record):
    output = {}
    for key, value in record.items():
        if key in {"match_info", "candidate_loc_lat_lon"}:
            continue
        if isinstance(value, (np.floating, np.integer)):
            output[key] = value.item()
        elif isinstance(value, tuple):
            output[key] = list(value)
        else:
            output[key] = value
    return output


def _extract_match_info(matcher):
    if matcher is None:
        return None
    if hasattr(matcher, "get_last_match_info"):
        return matcher.get_last_match_info()
    return getattr(matcher, "last_match_info", None)


def _should_retry_sparse_with_secondary(match_info):
    if not isinstance(match_info, dict):
        return True
    if bool(match_info.get("fallback_to_center", False)):
        return True
    if bool(match_info.get("identity_h_fallback", False)):
        return True
    reason = str(match_info.get("fallback_reason", "")).strip().lower()
    return reason in {"all_failed", "low_match", "low_quality", "projection_invalid", "out_of_bounds"}


def _accept_sparse_secondary_result(match_info, min_inliers=0, min_inlier_ratio=0.0):
    if _should_retry_sparse_with_secondary(match_info):
        return False
    if not isinstance(match_info, dict):
        return False
    inliers = int(match_info.get("inliers", 0))
    inlier_ratio = float(match_info.get("inlier_ratio", 0.0))
    if inliers < int(min_inliers):
        return False
    if inlier_ratio < float(min_inlier_ratio):
        return False
    return True


def _run_match_with_optional_secondary(
    primary_matcher,
    secondary_matcher,
    image0,
    image1,
    center_xy0,
    tl_xy0,
    yaw0=None,
    yaw1=None,
    rotate=True,
    case_name=None,
    secondary_accept_min_inliers=0,
    secondary_accept_min_inlier_ratio=0.0,
):
    primary_loc = primary_matcher.est_center(
        image0,
        image1,
        center_xy0,
        tl_xy0,
        yaw0=yaw0,
        yaw1=yaw1,
        rotate=rotate,
        case_name=case_name,
    )
    primary_info = _extract_match_info(primary_matcher)
    chosen_loc = primary_loc
    chosen_info = dict(primary_info) if isinstance(primary_info, dict) else primary_info
    chosen_matcher = primary_matcher
    secondary_retry_used = False

    if secondary_matcher is not None and _should_retry_sparse_with_secondary(primary_info):
        secondary_case_name = None if case_name is None else f"{case_name}_secondary"
        secondary_loc = secondary_matcher.est_center(
            image0,
            image1,
            center_xy0,
            tl_xy0,
            yaw0=yaw0,
            yaw1=yaw1,
            rotate=rotate,
            case_name=secondary_case_name,
            save_final_vis=False,
        )
        secondary_info = _extract_match_info(secondary_matcher)
        if _accept_sparse_secondary_result(
            secondary_info,
            min_inliers=secondary_accept_min_inliers,
            min_inlier_ratio=secondary_accept_min_inlier_ratio,
        ):
            chosen_loc = secondary_loc
            chosen_info = dict(secondary_info) if isinstance(secondary_info, dict) else secondary_info
            chosen_matcher = secondary_matcher
            secondary_retry_used = True

    if isinstance(chosen_info, dict):
        chosen_info = dict(chosen_info)
        chosen_info["secondary_retry_used"] = bool(secondary_retry_used)
        chosen_info["primary_fallback_reason"] = None if not isinstance(primary_info, dict) else primary_info.get("fallback_reason")

    return chosen_loc, chosen_info, chosen_matcher


def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    with torch.no_grad():
        
        for img in bar:
                    
            with autocast():
            
                img = img.to(train_config.device)
                img_feature = model(img1=img)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))

        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        
    if train_config.verbose:
        bar.close()
    
    return img_features


def evaluate(
        config,
        model,
        query_loader,
        gallery_loader,
        query_list,
        query_center_loc_xy_list,
        gallery_list,
        gallery_center_loc_xy_list,
        gallery_topleft_loc_xy_list,
        pairs_dict,
        query_yaw_list=None,
        ranks_list=[1, 5, 10],
        sdmk_list=[1, 3, 5],
        disk_list=[1, 3, 5],
        step_size=1000,
        cleanup=True,
        dis_threshold_list=[4*(i+1) for i in range(50)],
        plot_acc_threshold=False,
        top10_log=False,
        with_match=False,
        match_mode='sparse',
        rotate=True,
        sparse_angle_score_inlier_offset=None,
        sparse_use_multi_scale=True,
        sparse_scales=None,
        sparse_multi_scale_mode="both",
        sparse_allow_upsample=False,
        sparse_cross_scale_dedup_radius=0.0,
        sparse_lightglue_profile="current",
        sparse_sp_detection_threshold=0.0003,
        sparse_sp_max_num_keypoints=4096,
        sparse_sp_nms_radius=4,
        sparse_ransac_method="RANSAC",
        sparse_secondary_on_fallback=False,
        sparse_secondary_ransac_method="RANSAC",
        sparse_secondary_mode="per_candidate",
        sparse_secondary_accept_min_inliers=0,
        sparse_secondary_accept_min_inlier_ratio=0.0,
        sparse_ransac_reproj_threshold=20.0,
        sparse_min_inliers=15,
        sparse_min_inlier_ratio=0.001,
        sparse_save_final_vis=False,
        angle_experiment=False,
        orientation_checkpoint="",
        orientation_mode="off",
        orientation_fusion_weight=0.5,
        orientation_topk=1,
        confidence_checkpoint="",
        confidence_threshold=0.5,
        confidence_dump_path="",
        wandb_run=None,
        epoch=None,
        logger=None,
    ):
    ma_thresholds_m = (3.0, 5.0, 10.0, 20.0)
    t_total = time.perf_counter()
    if logger is not None:
        logger.info("开始评估：提取特征并计算匹配分数")
        logger.debug(
            "评估参数：ranks=%s, sdmk=%s, disk=%s, step_size=%s, with_match=%s, match_mode=%s, rotate=%s, sparse_angle_score_inlier_offset=%s, sparse_use_multi_scale=%s, sparse_scales=%s, sparse_multi_scale_mode=%s, sparse_allow_upsample=%s, sparse_cross_scale_dedup_radius=%.2f, sparse_lightglue_profile=%s, sparse_sp_det=%.6f, sparse_sp_max_kpts=%d, sparse_sp_nms=%d, sparse_ransac_method=%s, sparse_secondary_on_fallback=%s, sparse_secondary_ransac_method=%s, sparse_secondary_mode=%s, sparse_secondary_accept_min_inliers=%d, sparse_secondary_accept_min_inlier_ratio=%.6f, sparse_ransac_thresh=%.3f, sparse_min_inliers=%d, sparse_min_inlier_ratio=%.6f, sparse_save_final_vis=%s, angle_experiment=%s",
            ranks_list, sdmk_list, disk_list, step_size, with_match, match_mode, rotate, sparse_angle_score_inlier_offset, sparse_use_multi_scale, sparse_scales, sparse_multi_scale_mode, sparse_allow_upsample, float(sparse_cross_scale_dedup_radius), sparse_lightglue_profile, float(sparse_sp_detection_threshold), int(sparse_sp_max_num_keypoints), int(sparse_sp_nms_radius), sparse_ransac_method, bool(sparse_secondary_on_fallback), sparse_secondary_ransac_method, sparse_secondary_mode, int(sparse_secondary_accept_min_inliers), float(sparse_secondary_accept_min_inlier_ratio), float(sparse_ransac_reproj_threshold), int(sparse_min_inliers), float(sparse_min_inlier_ratio), sparse_save_final_vis, angle_experiment,
        )
    else:
        print("Extract Features and Compute Scores:")
    model.eval()
    t_query = time.perf_counter()
    img_features_query = predict(config, model, query_loader)
    query_extract_time = time.perf_counter() - t_query
    # img_features_gallery = predict(config, model, gallery_loader)

    all_scores = []
    t_gallery = time.perf_counter()
    with torch.no_grad():
        for gallery_batch in gallery_loader:
            with autocast():
                gallery_batch = gallery_batch.to(device=config.device)
                gallery_features_batch = model(img2=gallery_batch)
                if config.normalize_features:
                    gallery_features_batch = F.normalize(gallery_features_batch, dim=-1)

            scores_batch = img_features_query @ gallery_features_batch.T
            all_scores.append(scores_batch.cpu())
    gallery_infer_time = time.perf_counter() - t_gallery
    
    t_concat = time.perf_counter()
    all_scores = torch.cat(all_scores, dim=1).numpy()
    score_concat_time = time.perf_counter() - t_concat

    # with image match for finer loc
    if with_match:
        matcher = GimDKM(
            device=config.device,
            match_mode=match_mode,
            logger=logger,
            sparse_angle_score_inlier_offset=sparse_angle_score_inlier_offset,
            sparse_use_multi_scale=sparse_use_multi_scale,
            sparse_scales=sparse_scales,
            sparse_multi_scale_mode=sparse_multi_scale_mode,
            sparse_allow_upsample=sparse_allow_upsample,
            sparse_cross_scale_dedup_radius=sparse_cross_scale_dedup_radius,
            sparse_lightglue_profile=sparse_lightglue_profile,
            sparse_sp_detection_threshold=sparse_sp_detection_threshold,
            sparse_sp_max_num_keypoints=sparse_sp_max_num_keypoints,
            sparse_sp_nms_radius=sparse_sp_nms_radius,
            sparse_ransac_method=sparse_ransac_method,
            sparse_ransac_reproj_threshold=sparse_ransac_reproj_threshold,
            sparse_min_inliers=sparse_min_inliers,
            sparse_min_inlier_ratio=sparse_min_inlier_ratio,
            sparse_save_final_vis=sparse_save_final_vis,
        )
    secondary_matcher = None
    if (
        with_match
        and match_mode == "sparse"
        and bool(sparse_secondary_on_fallback)
    ):
        secondary_matcher = GimDKM(
            device=config.device,
            match_mode=match_mode,
            logger=logger,
            sparse_angle_score_inlier_offset=sparse_angle_score_inlier_offset,
            sparse_use_multi_scale=sparse_use_multi_scale,
            sparse_scales=sparse_scales,
            sparse_multi_scale_mode=sparse_multi_scale_mode,
            sparse_allow_upsample=sparse_allow_upsample,
            sparse_cross_scale_dedup_radius=sparse_cross_scale_dedup_radius,
            sparse_lightglue_profile=sparse_lightglue_profile,
            sparse_sp_detection_threshold=sparse_sp_detection_threshold,
            sparse_sp_max_num_keypoints=sparse_sp_max_num_keypoints,
            sparse_sp_nms_radius=sparse_sp_nms_radius,
            sparse_ransac_method=sparse_secondary_ransac_method,
            sparse_ransac_reproj_threshold=sparse_ransac_reproj_threshold,
            sparse_min_inliers=sparse_min_inliers,
            sparse_min_inlier_ratio=sparse_min_inlier_ratio,
            sparse_save_final_vis=False,
        )
    orientation_mode_key = str(orientation_mode).lower()
    orientation_model = None
    use_orientation_model = bool(
        with_match
        and match_mode == "sparse"
        and orientation_mode_key != "off"
        and str(orientation_checkpoint).strip()
    )
    if use_orientation_model:
        orientation_model = load_vop_checkpoint(orientation_checkpoint, device=config.device)
        if logger is not None:
            logger.info("VOP 已加载: %s", orientation_checkpoint)
    elif with_match and match_mode == "sparse" and orientation_mode_key != "off" and logger is not None:
        logger.warning("VOP 已请求但未提供有效 checkpoint，自动退回 geometry-only 选角。")
    confidence_verifier = None
    use_confidence_verifier = bool(
        with_match
        and match_mode == "sparse"
        and orientation_mode_key == "prior_topk"
        and str(confidence_checkpoint).strip()
    )
    if use_confidence_verifier:
        confidence_verifier = load_confidence_verifier(confidence_checkpoint)
        if logger is not None:
            logger.info("Confidence verifier 已加载: %s", confidence_checkpoint)
    elif str(confidence_checkpoint).strip() and logger is not None:
        logger.warning("Confidence verifier 已提供，但当前模式不是 prior_topk，自动忽略。")

    ap = 0.0

    gallery_idx = {}
    gallery_mapi_idx = {}
    for idx, gallery_img in enumerate(gallery_list):
        gallery_idx[gallery_img] = idx
        str_i = gallery_img.split('_')[0]
        gallery_mapi_idx.setdefault(str_i, []).append(idx)
    for k, v in gallery_mapi_idx.items():
        array = np.zeros(len(gallery_list), dtype=int)
        array[v] = 1
        gallery_mapi_idx[k] = array

    matches_list = []
    for query_i in query_list:
        pairs_list_i = pairs_dict[query_i]
        matches_i = []
        for pair in pairs_list_i:
            matches_i.append(gallery_idx[pair])
        matches_list.append(np.array(matches_i))

    matches_tensor = [torch.tensor(matches, dtype=torch.long) for matches in matches_list]

    query_num = img_features_query.shape[0]
    if logger is not None and with_match and match_mode == "sparse":
        if query_yaw_list is None:
            logger.warning("sparse yaw 检查: query_yaw_list=None，将无法应用方向先验 (可能因 --use_yaw 未启用或数据未提供)")
        else:
            valid_yaws = []
            invalid_count = 0
            for yaw in query_yaw_list:
                if yaw is None:
                    invalid_count += 1
                    continue
                try:
                    yaw_float = float(yaw)
                except (TypeError, ValueError):
                    invalid_count += 1
                    continue
                if not np.isfinite(yaw_float):
                    invalid_count += 1
                    continue
                valid_yaws.append(yaw_float)

            if len(valid_yaws) == 0:
                logger.warning(
                    "sparse yaw 检查: 全部无效，总数=%d 无效=%d",
                    len(query_yaw_list),
                    invalid_count,
                )
            else:
                yaw_arr = np.asarray(valid_yaws, dtype=np.float32)
                near_zero_ratio = float(np.mean(np.abs(yaw_arr) < 1e-3))
                logger.info(
                    "sparse yaw 统计: 总数=%d 有效=%d 无效=%d min=%.3f max=%.3f mean=%.3f near_zero_ratio=%.3f sample=%s",
                    len(query_yaw_list),
                    int(yaw_arr.shape[0]),
                    invalid_count,
                    float(np.min(yaw_arr)),
                    float(np.max(yaw_arr)),
                    float(np.mean(yaw_arr)),
                    near_zero_ratio,
                    str([round(float(v), 3) for v in yaw_arr[:5]]),
                )
                if near_zero_ratio > 0.9:
                    logger.warning("sparse yaw 检查: 超过90%%样本接近0度，建议确认 yaw 字段映射是否正确")

    all_ap = []
    cmc = np.zeros(len(gallery_list))
    sdm_list = []
    dis_list = []
    acc_threshold = [0 for _ in range(len(dis_threshold_list))]

    # for log
    top10_list = []
    loc1_list = []
    dis_ori_list = []
    dis_match_list = []
    orientation_stats = {
        "count": 0,
        "entropy_sum": 0.0,
        "concentration_sum": 0.0,
        "top_prob_sum": 0.0,
        "time_sum": 0.0,
        "match_time_sum": 0.0,
        "hypothesis_sum": 0,
        "override_count": 0,
        "improve_count": 0,
        "worse_count": 0,
    }
    confidence_stats = {
        "count": 0,
        "time_sum": 0.0,
        "accept_count": 0,
        "reject_count": 0,
        "override_count": 0,
        "selected_rank_sum": 0.0,
        "selected_score_sum": 0.0,
    }
    fineloc_stats = {
        "final_error_list": [],
        "coarse_error_list": [],
        "ma_hits": {float(thresh): 0 for thresh in ma_thresholds_m},
        "worse_than_coarse": 0,
        "fallback_count": 0,
        "identity_h_fallback_count": 0,
        "out_of_bounds_count": 0,
        "projection_invalid_count": 0,
        "retained_matches": [],
        "inliers": [],
        "inlier_ratios": [],
        "hypotheses_evaluated": [],
        "secondary_takeover_count": 0,
    }
    confidence_dump_records = [] if str(confidence_dump_path).strip() else None

    t_metrics = time.perf_counter()
    progress_interval = max(1, query_num // 10)
    for i in tqdm(range(query_num), desc="Processing each query"):
        str_i = query_list[i].split('_')[0]
        score = all_scores[i] * gallery_mapi_idx[str_i]
        # predict index
        index = np.argsort(score)[::-1]
        top1_index = index[0]

        # with image match for finer loc
        # match_loc: (lat, lon)
        # matcher.est_center (x, y) -> (lon, lat)
        match_loc = None
        match_info = None
        angle_results = []
        orientation_posterior = None
        matcher_used = matcher if with_match else None
        sparse_yaw0 = None
        sparse_yaw1 = None
        hypotheses_evaluated = 0
        if with_match:
            gallery_sample = gallery_loader.dataset[top1_index]
            query_sample = query_loader.dataset[i]
            if match_mode == "sparse":
                if query_yaw_list is not None and i < len(query_yaw_list) and query_yaw_list[i] is not None:
                    # D2S: satellite is treated as north-up (yaw=0), drone yaw comes from metadata.
                    sparse_yaw0 = 0.0
                    sparse_yaw1 = query_yaw_list[i]
            gallery_center_lat, gallery_center_lon = gallery_center_loc_xy_list[top1_index]
            gallery_center_lon_lat = gallery_center_lon, gallery_center_lat
            gallery_topleft_lat, gallery_topleft_lon = gallery_topleft_loc_xy_list[top1_index]
            gallery_topleft_lon_lat = gallery_topleft_lon, gallery_topleft_lat
            if use_orientation_model and orientation_mode_key == "prior_single":
                candidate_angles = getattr(orientation_model, "candidate_angles_deg", None) or [0.0]
                t_orientation = time.perf_counter()
                orientation_posterior = orientation_model.predict_posterior(
                    retrieval_model=model,
                    gallery_img=gallery_sample,
                    query_img=query_sample,
                    candidate_angles_deg=candidate_angles,
                    device=config.device,
                    gallery_branch="img2",
                    query_branch="img1",
                )
                orientation_stats["time_sum"] += time.perf_counter() - t_orientation
                rotated_query_sample = _rotate_query_tensor(query_sample, orientation_posterior["top_angle_deg"])
                t_match = time.perf_counter()
                match_loc_lon_lat, match_info, matcher_used = _run_match_with_optional_secondary(
                    matcher,
                    secondary_matcher,
                    gallery_sample,
                    rotated_query_sample,
                    gallery_center_lon_lat,
                    gallery_topleft_lon_lat,
                    yaw0=None,
                    yaw1=None,
                    rotate=0.0,
                    case_name=f"{query_list[i]}_vop_prior_single",
                    secondary_accept_min_inliers=sparse_secondary_accept_min_inliers,
                    secondary_accept_min_inlier_ratio=sparse_secondary_accept_min_inlier_ratio,
                )
                orientation_stats["match_time_sum"] += time.perf_counter() - t_match
                orientation_stats["hypothesis_sum"] += 1
                hypotheses_evaluated = 1
                orientation_stats["count"] += 1
                orientation_stats["entropy_sum"] += float(orientation_posterior["entropy"])
                orientation_stats["concentration_sum"] += float(orientation_posterior["concentration"])
                orientation_stats["top_prob_sum"] += float(orientation_posterior["top_prob"])
            elif use_orientation_model and orientation_mode_key == "prior_topk":
                candidate_angles = getattr(orientation_model, "candidate_angles_deg", None) or [0.0]
                topk = max(1, min(int(orientation_topk), len(candidate_angles)))
                t_orientation = time.perf_counter()
                orientation_posterior = orientation_model.predict_posterior(
                    retrieval_model=model,
                    gallery_img=gallery_sample,
                    query_img=query_sample,
                    candidate_angles_deg=candidate_angles,
                    device=config.device,
                    gallery_branch="img2",
                    query_branch="img1",
                )
                orientation_stats["time_sum"] += time.perf_counter() - t_orientation
                sorted_indices = list(np.argsort(np.asarray(orientation_posterior["probs"], dtype=np.float64))[::-1][:topk])
                best_match_loc_lon_lat = None
                best_match_info = None
                best_match_record = None
                best_matcher_used = matcher
                best_match_angle = None
                best_match_inliers = -1
                best_match_ratio = -1.0
                total_topk_match_time = 0.0
                coarse_error_m = float(get_dis_target(query_center_loc_xy_list[i], gallery_center_loc_xy_list[top1_index]))
                candidate_records = []
                for rank_j, cand_index in enumerate(sorted_indices):
                    cand_angle = float(candidate_angles[int(cand_index)])
                    cand_prob = float(orientation_posterior["probs"][int(cand_index)])
                    rotated_query_sample = _rotate_query_tensor(query_sample, cand_angle)
                    t_match = time.perf_counter()
                    if bool(sparse_secondary_on_fallback) and str(sparse_secondary_mode).lower() == "per_candidate":
                        candidate_loc_lon_lat, candidate_match_info, candidate_matcher_used = _run_match_with_optional_secondary(
                            matcher,
                            secondary_matcher,
                            gallery_sample,
                            rotated_query_sample,
                            gallery_center_lon_lat,
                            gallery_topleft_lon_lat,
                            yaw0=None,
                            yaw1=None,
                            rotate=0.0,
                            case_name=f"{query_list[i]}_vop_prior_top{topk}_{rank_j}_{cand_angle:.1f}",
                            secondary_accept_min_inliers=sparse_secondary_accept_min_inliers,
                            secondary_accept_min_inlier_ratio=sparse_secondary_accept_min_inlier_ratio,
                        )
                    else:
                        candidate_loc_lon_lat = matcher.est_center(
                            gallery_sample,
                            rotated_query_sample,
                            gallery_center_lon_lat,
                            gallery_topleft_lon_lat,
                            yaw0=None,
                            yaw1=None,
                            rotate=0.0,
                            case_name=f"{query_list[i]}_vop_prior_top{topk}_{rank_j}_{cand_angle:.1f}",
                        )
                        candidate_match_info = _extract_match_info(matcher) or {}
                        candidate_matcher_used = matcher
                    total_topk_match_time += time.perf_counter() - t_match
                    candidate_match_info = candidate_match_info or {}
                    cand_kept = int(candidate_match_info.get("n_kept", 0))
                    cand_inliers = int(candidate_match_info.get("inliers", 0))
                    cand_ratio = float(cand_inliers) / float(max(cand_kept, 1))
                    candidate_loc_lat_lon = None
                    candidate_error_m = None
                    if candidate_loc_lon_lat is not None:
                        candidate_loc_lat_lon = (candidate_loc_lon_lat[1], candidate_loc_lon_lat[0])
                        candidate_error_m = float(geodesic(query_center_loc_xy_list[i], candidate_loc_lat_lon).meters)
                    candidate_record = build_confidence_candidate_record(
                        candidate_rank=rank_j,
                        candidate_angle_deg=cand_angle,
                        candidate_prob=cand_prob,
                        top_prob=float(orientation_posterior["top_prob"]),
                        entropy=float(orientation_posterior["entropy"]),
                        concentration=float(orientation_posterior["concentration"]),
                        match_info=candidate_match_info,
                        coarse_error_m=coarse_error_m,
                        candidate_error_m=candidate_error_m,
                        score_offset=sparse_angle_score_inlier_offset,
                    )
                    candidate_record["candidate_loc_lat_lon"] = candidate_loc_lat_lon
                    candidate_record["match_info"] = dict(candidate_match_info)
                    candidate_records.append(candidate_record)
                    if _prefer_match_candidate(cand_inliers, cand_ratio, best_match_inliers, best_match_ratio):
                        best_match_loc_lon_lat = candidate_loc_lon_lat
                        best_match_info = dict(candidate_match_info)
                        best_match_record = dict(candidate_record)
                        best_matcher_used = candidate_matcher_used
                        best_match_angle = cand_angle
                        best_match_inliers = cand_inliers
                        best_match_ratio = cand_ratio
                if (
                    bool(sparse_secondary_on_fallback)
                    and str(sparse_secondary_mode).lower() == "final_only"
                    and secondary_matcher is not None
                    and _should_retry_sparse_with_secondary(best_match_info)
                    and best_match_angle is not None
                ):
                    primary_fallback_reason = None if not isinstance(best_match_info, dict) else best_match_info.get("fallback_reason")
                    rotated_query_sample = _rotate_query_tensor(query_sample, best_match_angle)
                    secondary_loc_lon_lat = secondary_matcher.est_center(
                        gallery_sample,
                        rotated_query_sample,
                        gallery_center_lon_lat,
                        gallery_topleft_lon_lat,
                        yaw0=None,
                        yaw1=None,
                        rotate=0.0,
                        case_name=f"{query_list[i]}_vop_prior_top{topk}_final_secondary_{best_match_angle:.1f}",
                        save_final_vis=False,
                    )
                    secondary_match_info = _extract_match_info(secondary_matcher) or {}
                    if _accept_sparse_secondary_result(
                        secondary_match_info,
                        min_inliers=sparse_secondary_accept_min_inliers,
                        min_inlier_ratio=sparse_secondary_accept_min_inlier_ratio,
                    ):
                        best_match_loc_lon_lat = None
                        if secondary_loc_lon_lat is not None:
                            best_match_loc_lon_lat = secondary_loc_lon_lat
                        best_match_info = dict(secondary_match_info)
                        best_match_info["secondary_retry_used"] = True
                        best_match_info["primary_fallback_reason"] = primary_fallback_reason
                        best_matcher_used = secondary_matcher
                match_loc_lon_lat = best_match_loc_lon_lat
                match_info = best_match_info
                matcher_used = best_matcher_used
                if use_confidence_verifier and len(candidate_records) > 0:
                    t_conf = time.perf_counter()
                    selected_conf_record = select_candidate_with_confidence(candidate_records, confidence_verifier)
                    confidence_stats["time_sum"] += time.perf_counter() - t_conf
                    confidence_stats["count"] += 1
                    if selected_conf_record is not None:
                        confidence_score = float(selected_conf_record.get("confidence_score", 0.0))
                        confidence_stats["selected_score_sum"] += confidence_score
                        confidence_stats["selected_rank_sum"] += float(int(selected_conf_record.get("candidate_rank", 0)) + 1)
                        if best_match_record is not None and int(selected_conf_record.get("candidate_rank", -1)) != int(best_match_record.get("candidate_rank", -1)):
                            confidence_stats["override_count"] += 1
                        if bool(selected_conf_record.get("confidence_accept", False)) and selected_conf_record.get("candidate_loc_lat_lon") is not None:
                            confidence_stats["accept_count"] += 1
                            selected_loc_lat_lon = tuple(selected_conf_record["candidate_loc_lat_lon"])
                            match_loc_lon_lat = (selected_loc_lat_lon[1], selected_loc_lat_lon[0])
                            match_info = dict(selected_conf_record.get("match_info", {}))
                        else:
                            confidence_stats["reject_count"] += 1
                            coarse_lat_lon = tuple(gallery_center_loc_xy_list[top1_index])
                            match_loc_lon_lat = (coarse_lat_lon[1], coarse_lat_lon[0])
                            match_info = dict(best_match_info) if isinstance(best_match_info, dict) else {}
                            match_info["fallback_to_center"] = True
                            match_info["fallback_reason"] = "confidence_reject"
                            match_info["confidence_reject"] = True
                        if isinstance(match_info, dict):
                            match_info = dict(match_info)
                            match_info["confidence_score"] = confidence_score
                            match_info["confidence_accept"] = bool(selected_conf_record.get("confidence_accept", False))
                            match_info["confidence_rank"] = int(selected_conf_record.get("candidate_rank", 0)) + 1
                            match_info["confidence_candidate_prob"] = float(selected_conf_record.get("candidate_prob", 0.0))
                            match_info["confidence_geometry_score"] = float(selected_conf_record.get("geometry_score", 0.0))
                if confidence_dump_records is not None:
                    confidence_dump_records.append(
                        {
                            "query_name": query_list[i],
                            "top1_gallery_name": gallery_list[top1_index],
                            "coarse_error_m": coarse_error_m,
                            "geometry_selected_rank": None if best_match_record is None else int(best_match_record.get("candidate_rank", 0)) + 1,
                            "geometry_selected_error_m": None if best_match_record is None else best_match_record.get("candidate_error_m"),
                            "candidates": [_to_jsonable_candidate_record(record) for record in candidate_records],
                        }
                    )
                orientation_stats["match_time_sum"] += total_topk_match_time
                orientation_stats["hypothesis_sum"] += topk
                hypotheses_evaluated = topk
                orientation_stats["count"] += 1
                orientation_stats["entropy_sum"] += float(orientation_posterior["entropy"])
                orientation_stats["concentration_sum"] += float(orientation_posterior["concentration"])
                orientation_stats["top_prob_sum"] += float(orientation_posterior["top_prob"])
            else:
                t_match = time.perf_counter()
                match_loc_lon_lat, match_info, matcher_used = _run_match_with_optional_secondary(
                    matcher,
                    secondary_matcher,
                    gallery_sample,
                    query_sample,
                    gallery_center_lon_lat,
                    gallery_topleft_lon_lat,
                    yaw0=sparse_yaw0,
                    yaw1=sparse_yaw1,
                    rotate=rotate,
                    case_name=query_list[i],
                    secondary_accept_min_inliers=sparse_secondary_accept_min_inliers,
                    secondary_accept_min_inlier_ratio=sparse_secondary_accept_min_inlier_ratio,
                )
                orientation_stats["match_time_sum"] += time.perf_counter() - t_match
            if match_info is None:
                match_info = _extract_match_info(matcher_used)
            if matcher_used is not None and hasattr(matcher_used, "get_last_angle_results"):
                angle_results = matcher_used.get_last_angle_results()
            if hypotheses_evaluated <= 0:
                hypotheses_evaluated = max(
                    1,
                    sum(1 for angle_result in angle_results if str(angle_result.get("status", "")) == "ok"),
                )
            match_loc_lat_lon = (match_loc_lon_lat[1], match_loc_lon_lat[0])
            match_loc = match_loc_lat_lon

            if use_orientation_model and orientation_mode_key not in {"off", "prior_single", "prior_topk"} and len(angle_results) > 0:
                candidate_angles = _build_unique_candidate_angles(angle_results)
                if candidate_angles:
                    t_orientation = time.perf_counter()
                    orientation_posterior = orientation_model.predict_posterior(
                        retrieval_model=model,
                        gallery_img=gallery_sample,
                        query_img=query_sample,
                        candidate_angles_deg=candidate_angles,
                        device=config.device,
                        gallery_branch="img2",
                        query_branch="img1",
                    )
                    orientation_stats["time_sum"] += time.perf_counter() - t_orientation
                    selected_angle_result = select_angle_result_with_vop(
                        angle_results,
                        posterior=orientation_posterior,
                        mode=str(orientation_mode).lower(),
                        fusion_weight=orientation_fusion_weight,
                    )
                    orientation_stats["count"] += 1
                    orientation_stats["entropy_sum"] += float(orientation_posterior["entropy"])
                    orientation_stats["concentration_sum"] += float(orientation_posterior["concentration"])
                    orientation_stats["top_prob_sum"] += float(orientation_posterior["top_prob"])
                    if selected_angle_result is not None:
                        selected_loc_lon_lat = project_match_center_from_h(
                            selected_angle_result.get("homography"),
                            gallery_sample,
                            gallery_center_lon_lat,
                            gallery_topleft_lon_lat,
                        )
                        if selected_loc_lon_lat is not None:
                            selected_loc_lat_lon = (selected_loc_lon_lat[1], selected_loc_lon_lat[0])
                            baseline_distance = geodesic(query_center_loc_xy_list[i], match_loc_lat_lon).meters
                            selected_distance = geodesic(query_center_loc_xy_list[i], selected_loc_lat_lon).meters
                            if (
                                abs(float(selected_loc_lat_lon[0]) - float(match_loc_lat_lon[0])) > 1e-9
                                or abs(float(selected_loc_lat_lon[1]) - float(match_loc_lat_lon[1])) > 1e-9
                            ):
                                orientation_stats["override_count"] += 1
                            if selected_distance + 1e-6 < baseline_distance:
                                orientation_stats["improve_count"] += 1
                            elif selected_distance > baseline_distance + 1e-6:
                                orientation_stats["worse_count"] += 1
                            match_loc = selected_loc_lat_lon
                            if isinstance(match_info, dict):
                                match_info = dict(match_info)
                                match_info["final_vis_path"] = None
            dis_match_list.append(get_dis_target(query_center_loc_xy_list[i], match_loc))
        else:
            dis_match_list.append(None)
        
        dis_ori_list.append(get_dis_target(query_center_loc_xy_list[i], gallery_center_loc_xy_list[top1_index]))

        sdm_list.append(sdm(query_center_loc_xy_list[i], sdmk_list, index, gallery_center_loc_xy_list))

        dis_list.append(get_dis(query_center_loc_xy_list[i], index, gallery_center_loc_xy_list, disk_list, match_loc))
        if with_match:
            final_error_m = float(dis_list[i][0])
            coarse_error_m = float(dis_ori_list[i])
            fineloc_stats["final_error_list"].append(final_error_m)
            fineloc_stats["coarse_error_list"].append(coarse_error_m)
            for thresh in ma_thresholds_m:
                if final_error_m <= float(thresh):
                    fineloc_stats["ma_hits"][float(thresh)] += 1
            if final_error_m > coarse_error_m + 1e-6:
                fineloc_stats["worse_than_coarse"] += 1
            fineloc_stats["hypotheses_evaluated"].append(float(hypotheses_evaluated))
            if isinstance(match_info, dict):
                retained_matches = float(match_info.get("n_kept", 0))
                inliers = float(match_info.get("inliers", 0))
                inlier_ratio = float(match_info.get("inlier_ratio", 0.0))
                if retained_matches <= 0 and inliers > 0:
                    retained_matches = inliers
                if retained_matches > 0 and inlier_ratio <= 0:
                    inlier_ratio = float(inliers) / float(max(retained_matches, 1.0))
                fineloc_stats["retained_matches"].append(retained_matches)
                fineloc_stats["inliers"].append(inliers)
                fineloc_stats["inlier_ratios"].append(inlier_ratio)
                if bool(match_info.get("fallback_to_center", False)):
                    fineloc_stats["fallback_count"] += 1
                if bool(match_info.get("identity_h_fallback", False)):
                    fineloc_stats["identity_h_fallback_count"] += 1
                if bool(match_info.get("out_of_bounds", False)):
                    fineloc_stats["out_of_bounds_count"] += 1
                if bool(match_info.get("projection_invalid", False)):
                    fineloc_stats["projection_invalid_count"] += 1
                if bool(match_info.get("secondary_retry_used", False)):
                    fineloc_stats["secondary_takeover_count"] += 1
        if with_match and isinstance(match_info, dict):
            annotate_final_match_visualization(
                match_info.get("final_vis_path"),
                dis_list[i][0],
                logger=logger,
            )
        if logger is not None and angle_experiment and with_match and match_mode == "sparse":
            logger.info(
                "[角度实验] Query=%s | Top1Gallery=%s | 原始Top1Dis=%.2fm | 最终Dis=%.2fm | 候选角度数=%d",
                query_list[i],
                gallery_list[top1_index],
                float(dis_ori_list[i]),
                float(dis_list[i][0]),
                len(angle_results),
            )
            if orientation_posterior is not None:
                logger.info(
                    "[VOP] Query=%s | top_angle=%.1f° | top_prob=%.4f | entropy=%.4f | concentration=%.4f",
                    query_list[i],
                    float(orientation_posterior["top_angle_deg"]),
                    float(orientation_posterior["top_prob"]),
                    float(orientation_posterior["entropy"]),
                    float(orientation_posterior["concentration"]),
                )
            gallery_center_lat, gallery_center_lon = gallery_center_loc_xy_list[top1_index]
            gallery_center_lon_lat = gallery_center_lon, gallery_center_lat
            gallery_topleft_lat, gallery_topleft_lon = gallery_topleft_loc_xy_list[top1_index]
            gallery_topleft_lon_lat = gallery_topleft_lon, gallery_topleft_lat
            gallery_sample_for_exp = gallery_loader.dataset[top1_index]
            for angle_result in angle_results:
                angle_loc_lon_lat = project_match_center_from_h(
                    angle_result.get("homography"),
                    gallery_sample_for_exp,
                    gallery_center_lon_lat,
                    gallery_topleft_lon_lat,
                )
                if angle_loc_lon_lat is not None:
                    angle_loc_lat_lon = angle_loc_lon_lat[1], angle_loc_lon_lat[0]
                    angle_dis_text = f"{geodesic(query_center_loc_xy_list[i], angle_loc_lat_lon).meters:.2f}m"
                else:
                    angle_dis_text = "NA"
                logger.info(
                    "[角度实验] phase=%d | search=%.1f° | rot=%.1f° | matches=%d | inliers=%d | ratio=%.4f | score=%s | dist=%s | selected=%s | status=%s",
                    int(angle_result.get("phase", 0)),
                    float(angle_result.get("search_angle", 0.0)),
                    float(angle_result.get("rot_angle", 0.0)),
                    int(angle_result.get("matches", 0)),
                    int(angle_result.get("inliers", 0)),
                    float(angle_result.get("ratio", 0.0)),
                    "NA" if angle_result.get("score") is None else f"{float(angle_result.get('score', 0.0)):.4f}",
                    angle_dis_text,
                    "yes" if angle_result.get("selected") else "no",
                    angle_result.get("status", "unknown"),
                )
            logger.info("")
        if logger is not None and with_match:
            phase_value = None
            inliers_value = None
            if isinstance(match_info, dict):
                phase_value = match_info.get("phase")
                inliers_value = match_info.get("inliers")
            phase_str = f"第{int(phase_value)}阶段" if phase_value is not None else "未知阶段"
            inliers_str = str(int(inliers_value)) if inliers_value is not None else "未知"
            logger.debug(
                "样本总结: Query=%s | Top1Gallery=%s | 最后用了%s | 内点数=%s | 最终Dis=%.2fm\n",
                query_list[i],
                gallery_list[top1_index],
                phase_str,
                inliers_str,
                float(dis_list[i][0]),
            )
            if isinstance(match_info, dict):
                scale_stats = match_info.get("scale_stats", [])
                if scale_stats:
                    logger.debug(
                        "样本尺度贡献: %s",
                        " ; ".join(
                            "{}(q={:.2f},g={:.2f}): kept={} inliers={}".format(
                                str(item.get("label", "")),
                                float(item.get("q_scale", 0.0)),
                                float(item.get("g_scale", 0.0)),
                                int(item.get("retained_matches", 0)),
                                int(item.get("inliers", 0)),
                            )
                            for item in scale_stats
                        ),
                    )

        top10_list.append(get_top10(index, gallery_list))
        loc1_lat, loc1_lon = gallery_center_loc_xy_list[index[0]]
        if with_match:
            loc1_lat, loc1_lon = match_loc
        loc1_list.append((query_center_loc_xy_list[i][0], query_center_loc_xy_list[i][1], loc1_lat, loc1_lon))

        for j in range(len(dis_threshold_list)):
            if dis_list[i][0] < dis_threshold_list[j]:
                acc_threshold[j] += 1.

        good_index_i = np.isin(index, matches_tensor[i]) 

        # 计算 AP
        y_true = good_index_i.astype(int)
        y_scores = np.arange(len(y_true), 0, -1)  # 分数与排名相反
        if np.sum(y_true) > 0:  # 仅计算有正样本的情况
            ap = average_precision_score(y_true, y_scores)
            all_ap.append(ap)
        
        # 计算 CMC
        match_rank = np.where(good_index_i == 1)[0]
        if len(match_rank) > 0:
            cmc[match_rank[0]:] += 1
        if logger is not None and ((i + 1) % progress_interval == 0 or i == query_num - 1):
            logger.debug("评估进度：%d/%d (%.1f%%)", i + 1, query_num, (i + 1) * 100.0 / query_num)
    
    metrics_time = time.perf_counter() - t_metrics
    if with_match:
        matcher.summarize_and_log()
    mAP = np.mean(all_ap)
    cmc = cmc / query_num

    sdm_list = np.mean(np.array(sdm_list), axis=0)
    dis_list = np.mean(np.array(dis_list), axis=0)

    # top 1%
    top1 = round(len(gallery_list)*0.01)

    string = []

    for i in ranks_list:
        string.append('Recall@{}: {:.4f}'.format(i, cmc[i-1]*100))
        
    string.append('Recall@top1: {:.4f}'.format(cmc[top1]*100))
    string.append('AP: {:.4f}'.format(mAP*100))   
    
    for i in range(len(sdmk_list)):
        string.append('SDM@{}: {:.4f}'.format(sdmk_list[i], sdm_list[i]))
    for i in range(len(disk_list)):
        string.append('Dis@{}: {:.4f}'.format(disk_list[i], dis_list[i]))
    if with_match:
        final_error_arr = np.asarray(fineloc_stats["final_error_list"], dtype=np.float64)
        coarse_error_arr = np.asarray(fineloc_stats["coarse_error_list"], dtype=np.float64)
        retained_arr = np.asarray(fineloc_stats["retained_matches"], dtype=np.float64)
        inliers_arr = np.asarray(fineloc_stats["inliers"], dtype=np.float64)
        inlier_ratio_arr = np.asarray(fineloc_stats["inlier_ratios"], dtype=np.float64)
        hypotheses_arr = np.asarray(fineloc_stats["hypotheses_evaluated"], dtype=np.float64)
        query_count_for_stats = max(int(query_num), 1)
        ma_metrics = {
            int(thresh): float(fineloc_stats["ma_hits"][float(thresh)]) / float(query_count_for_stats) * 100.0
            for thresh in ma_thresholds_m
        }
        fallback_ratio = float(fineloc_stats["fallback_count"]) / float(query_count_for_stats) * 100.0
        worse_ratio = float(fineloc_stats["worse_than_coarse"]) / float(query_count_for_stats) * 100.0
        mean_retained = float(np.mean(retained_arr)) if retained_arr.size > 0 else 0.0
        mean_inliers = float(np.mean(inliers_arr)) if inliers_arr.size > 0 else 0.0
        mean_inlier_ratio = float(np.mean(inlier_ratio_arr)) if inlier_ratio_arr.size > 0 else 0.0
        mean_hypotheses = float(np.mean(hypotheses_arr)) if hypotheses_arr.size > 0 else 0.0
        mean_vop_time = float(orientation_stats["time_sum"]) / float(query_count_for_stats)
        mean_match_time = float(orientation_stats["match_time_sum"]) / float(query_count_for_stats)
        mean_confidence_time = float(confidence_stats["time_sum"]) / float(query_count_for_stats)
        mean_total_fineloc_time = mean_vop_time + mean_match_time + mean_confidence_time
        string.extend(
            [
                'MA@3m: {:.2f}'.format(ma_metrics[3]),
                'MA@5m: {:.2f}'.format(ma_metrics[5]),
                'MA@10m: {:.2f}'.format(ma_metrics[10]),
                'MA@20m: {:.2f}'.format(ma_metrics[20]),
            ]
        )

    result_str = ' - '.join(string)
    if logger is not None:
        logger.info(result_str)
        logger.info(
            "评估摘要：Recall@1=%.4f%%, Recall@5=%.4f%%, Recall@10=%.4f%%, AP=%.4f%%",
            cmc[0] * 100,
            cmc[4] * 100 if len(cmc) > 4 else -1.0,
            cmc[9] * 100 if len(cmc) > 9 else -1.0,
            mAP * 100,
        )
        if orientation_stats["count"] > 0:
            orientation_count = max(int(orientation_stats["count"]), 1)
            logger.info(
                "VOP 摘要：count=%d mean_top_prob=%.4f mean_entropy=%.4f mean_concentration=%.4f mean_hypotheses=%.2f override=%d improve=%d worse=%d",
                int(orientation_stats["count"]),
                orientation_stats["top_prob_sum"] / orientation_count,
                orientation_stats["entropy_sum"] / orientation_count,
                orientation_stats["concentration_sum"] / orientation_count,
                float(orientation_stats["hypothesis_sum"]) / orientation_count,
                int(orientation_stats["override_count"]),
                int(orientation_stats["improve_count"]),
                int(orientation_stats["worse_count"]),
            )
            logger.info(
                "VOP 耗时：mean_posterior_time=%.6fs/query mean_match_time=%.6fs/query mean_total_time=%.6fs/query",
                orientation_stats["time_sum"] / orientation_count,
                orientation_stats["match_time_sum"] / orientation_count,
                (orientation_stats["time_sum"] + orientation_stats["match_time_sum"]) / orientation_count,
            )
        if with_match:
            logger.info(
                "FineLoc 阈值成功率: MA@3m=%.2f%% MA@5m=%.2f%% MA@10m=%.2f%% MA@20m=%.2f%%",
                ma_metrics[3],
                ma_metrics[5],
                ma_metrics[10],
                ma_metrics[20],
            )
            logger.info(
                "FineLoc 稳健性统计: worse_than_coarse=%d(%.2f%%) fallback=%d(%.2f%%) identity_H_fallback=%d out_of_bounds=%d projection_invalid=%d",
                int(fineloc_stats["worse_than_coarse"]),
                worse_ratio,
                int(fineloc_stats["fallback_count"]),
                fallback_ratio,
                int(fineloc_stats["identity_h_fallback_count"]),
                int(fineloc_stats["out_of_bounds_count"]),
                int(fineloc_stats["projection_invalid_count"]),
            )
            logger.info(
                "FineLoc 匹配统计: mean_hypotheses=%.2f mean_retained_matches=%.1f mean_inliers=%.1f mean_inlier_ratio=%.4f",
                mean_hypotheses,
                mean_retained,
                mean_inliers,
                mean_inlier_ratio,
            )
            logger.info(
                "FineLoc 运行时间: mean_vop_forward=%.6fs/query mean_matcher=%.6fs/query mean_total=%.6fs/query",
                mean_vop_time,
                mean_match_time,
                mean_total_fineloc_time,
            )
            if bool(sparse_secondary_on_fallback):
                logger.info(
                    "FineLoc 双路径回退统计: secondary_takeover=%d(%.2f%%) primary_method=%s secondary_method=%s mode=%s accept_min_inliers=%d accept_min_inlier_ratio=%.4f",
                    int(fineloc_stats["secondary_takeover_count"]),
                    float(fineloc_stats["secondary_takeover_count"]) * 100.0 / float(max(query_num, 1)),
                    str(sparse_ransac_method),
                    str(sparse_secondary_ransac_method),
                    str(sparse_secondary_mode),
                    int(sparse_secondary_accept_min_inliers),
                    float(sparse_secondary_accept_min_inlier_ratio),
                )
            if confidence_stats["count"] > 0:
                confidence_count = max(int(confidence_stats["count"]), 1)
                logger.info(
                    "Confidence 验证统计: count=%d accept=%d reject=%d override=%d mean_selected_rank=%.2f mean_score=%.4f mean_extra_time=%.6fs/query threshold=%.3f",
                    int(confidence_stats["count"]),
                    int(confidence_stats["accept_count"]),
                    int(confidence_stats["reject_count"]),
                    int(confidence_stats["override_count"]),
                    float(confidence_stats["selected_rank_sum"]) / float(confidence_count),
                    float(confidence_stats["selected_score_sum"]) / float(confidence_count),
                    mean_confidence_time,
                    float(confidence_threshold),
                )
        logger.debug(
            "评估耗时统计：query提取=%.6fs, gallery推理=%.6fs, 分数拼接=%.6fs, 指标计算=%.6fs, 总耗时=%.6fs",
            query_extract_time,
            gallery_infer_time,
            score_concat_time,
            metrics_time,
            time.perf_counter() - t_total,
        )
    else:
        print(result_str)

    if wandb_run is not None:
        log_data = {
            "eval/recall@1": float(cmc[0] * 100),
            "eval/mAP": float(mAP * 100),
            "eval/sdm@1": float(sdm_list[0]) if len(sdm_list) > 0 else 0.0,
            "eval/dis@1": float(dis_list[0]) if len(dis_list) > 0 else 0.0,
            "time/eval_visloc/query_extract_s": query_extract_time,
            "time/eval_visloc/gallery_infer_s": gallery_infer_time,
            "time/eval_visloc/score_concat_s": score_concat_time,
            "time/eval_visloc/metrics_s": metrics_time,
            "time/eval_visloc/total_s": time.perf_counter() - t_total,
            "eval/query_num": int(query_num),
            "eval/gallery_num": int(len(gallery_list)),
        }
        if len(cmc) > 4:
            log_data["eval/recall@5"] = float(cmc[4] * 100)
        if len(cmc) > 9:
            log_data["eval/recall@10"] = float(cmc[9] * 100)
        if with_match:
            log_data["eval/ma@3m"] = ma_metrics[3]
            log_data["eval/ma@5m"] = ma_metrics[5]
            log_data["eval/ma@10m"] = ma_metrics[10]
            log_data["eval/ma@20m"] = ma_metrics[20]
            log_data["eval/worse_than_coarse_count"] = int(fineloc_stats["worse_than_coarse"])
            log_data["eval/worse_than_coarse_ratio"] = worse_ratio
            log_data["eval/fallback_count"] = int(fineloc_stats["fallback_count"])
            log_data["eval/fallback_ratio"] = fallback_ratio
            log_data["eval/identity_h_fallback_count"] = int(fineloc_stats["identity_h_fallback_count"])
            log_data["eval/out_of_bounds_count"] = int(fineloc_stats["out_of_bounds_count"])
            log_data["eval/projection_invalid_count"] = int(fineloc_stats["projection_invalid_count"])
            log_data["eval/mean_hypotheses"] = mean_hypotheses
            log_data["eval/mean_retained_matches"] = mean_retained
            log_data["eval/mean_inliers"] = mean_inliers
            log_data["eval/mean_inlier_ratio"] = mean_inlier_ratio
            log_data["time/eval_visloc/fine_loc_vop_mean_s"] = mean_vop_time
            log_data["time/eval_visloc/fine_loc_matcher_mean_s"] = mean_match_time
            log_data["time/eval_visloc/fine_loc_confidence_mean_s"] = mean_confidence_time
            log_data["time/eval_visloc/fine_loc_total_mean_s"] = mean_total_fineloc_time
        if orientation_stats["count"] > 0:
            orientation_count = max(int(orientation_stats["count"]), 1)
            log_data["eval/vop_mean_top_prob"] = float(orientation_stats["top_prob_sum"] / orientation_count)
            log_data["eval/vop_mean_entropy"] = float(orientation_stats["entropy_sum"] / orientation_count)
            log_data["eval/vop_mean_concentration"] = float(orientation_stats["concentration_sum"] / orientation_count)
            log_data["time/eval_visloc/vop_mean_s"] = float(orientation_stats["time_sum"] / orientation_count)
            log_data["time/eval_visloc/vop_match_mean_s"] = float(orientation_stats["match_time_sum"] / orientation_count)
            log_data["time/eval_visloc/vop_total_mean_s"] = float((orientation_stats["time_sum"] + orientation_stats["match_time_sum"]) / orientation_count)
            log_data["eval/vop_mean_hypotheses"] = float(orientation_stats["hypothesis_sum"] / orientation_count)
            log_data["eval/vop_override_count"] = int(orientation_stats["override_count"])
            log_data["eval/vop_improve_count"] = int(orientation_stats["improve_count"])
            log_data["eval/vop_worse_count"] = int(orientation_stats["worse_count"])
        if confidence_stats["count"] > 0:
            confidence_count = max(int(confidence_stats["count"]), 1)
            log_data["eval/confidence_accept_count"] = int(confidence_stats["accept_count"])
            log_data["eval/confidence_reject_count"] = int(confidence_stats["reject_count"])
            log_data["eval/confidence_override_count"] = int(confidence_stats["override_count"])
            log_data["eval/confidence_mean_selected_rank"] = float(confidence_stats["selected_rank_sum"]) / float(confidence_count)
            log_data["eval/confidence_mean_score"] = float(confidence_stats["selected_score_sum"]) / float(confidence_count)
            log_data["time/eval_visloc/confidence_mean_s"] = mean_confidence_time
        if epoch is not None:
            log_data["eval/epoch"] = int(epoch)
        wandb_run.log(log_data)

    if confidence_dump_records is not None:
        dump_path = os.path.abspath(str(confidence_dump_path))
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(confidence_dump_records, f, ensure_ascii=False, indent=2)
        if logger is not None:
            logger.info("Confidence dump 已保存: %s (queries=%d)", dump_path, len(confidence_dump_records))
    
    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, gallery_features_batch, scores_batch
        gc.collect()
        #torch.cuda.empty_cache()

    if top10_log:
        for query_img, top10, loc, dis_ori, dis_match in zip(query_list, top10_list, loc1_list, dis_ori_list, dis_match_list):
            print('Query', query_img)
            print('Top10', top10)
            print('Query loc', loc[0], loc[1])
            print('Top1 loc', loc[2], loc[3])
            
            imgs_type = []
            for img_name in top10[:5]:
                if img_name in pairs_dict[query_img]:
                    imgs_type.append('Pos')
                else:
                    imgs_type.append('Null')
            print(imgs_type)

            if dis_ori < dis_match:
                print(f'before match is better, dis_ori={dis_ori}, dis_match={dis_match}')


    if plot_acc_threshold:
        y = np.array(acc_threshold)
        x = np.array(dis_threshold_list)
        y = y / query_num * 100

        # x_new = np.linspace(x.min(), x.max(), 500)
        # spl = make_interp_spline(x, y, k=3)  
        # y_smooth = spl(x_new)

        # plt.figure(figsize=(10, 6), dpi=300)
        # plt.plot(x_new, y_smooth, label='Smooth Curve', color='red')
        # plt.scatter(x, y, label='Discrete Points', color='blue')

        # plt.xlabel('X Axis')
        # plt.ylabel('Y Axis')
        # plt.title('Smooth Curve with Discrete Points')
        # plt.legend()

        # # 调整边框
        # plt.gca().spines['top'].set_visible(False)
        # plt.gca().spines['right'].set_visible(False)

        # 显示图表
        # plt.tight_layout()
        # plt.savefig('/home/xmuairmud/jyx/GTA-UAV-private/Game4Loc/images/plot_acc_threshold_samearea.png')
        print(y.tolist())
    
    return cmc[0]
