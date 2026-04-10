import torch
import numpy as np
from tqdm import tqdm
import gc
import time
import logging
from collections import OrderedDict
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from sklearn.metrics import average_precision_score
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from PIL import Image, ImageDraw, ImageFont
import pickle
import os

from ..matcher.gim_dkm import GimDKM
from ..orientation import load_vop_checkpoint


def sdm(query_loc, sdmk_list, index, gallery_loc_xy_list, s=0.001):
    query_x, query_y = query_loc

    sdm_list = []

    for k in sdmk_list:
        sdm_nom = 0.0
        sdm_den = 0.0
        for i in range(k):
            idx = index[i]
            gallery_x, gallery_y = gallery_loc_xy_list[idx]
            d = np.sqrt((query_x - gallery_x)**2 + (query_y - gallery_y)**2)
            sdm_nom += (k - i) / np.exp(s * d) 
            sdm_den += (k - i)
        sdm_list.append(sdm_nom/sdm_den)
    return sdm_list


def get_dis(query_loc, index, gallery_loc_xy_list, disk_list, match_loc=None):
    query_x, query_y = query_loc
    dis_list = []
    for k in disk_list:
        dis_sum = 0.0
        for i in range(k):
            idx = index[i]
            gallery_x, gallery_y = gallery_loc_xy_list[idx]
            dis = np.sqrt((query_x - gallery_x)**2 + (query_y - gallery_y)**2)
            dis_sum += dis
        
        # For matcher estimated location
        if k == 1 and match_loc != None:
            match_loc_x, match_loc_y = match_loc
            dis_match = np.sqrt((query_x - match_loc_x)**2 + (query_y - match_loc_y)**2)

            dis_list.append(dis_match)
        else:
            dis_list.append(dis_sum / k)

    return dis_list


def get_dis_target(query_loc, target_loc):
    query_x, query_y = query_loc
    target_x, target_y = target_loc
    return np.sqrt((query_x-target_x)**2 + (query_y-target_y)**2)


def get_top10(index, gallery_list):
    top10 = []
    for i in range(10):
        idx = index[i]
        top10.append(gallery_list[idx])
    return top10


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


def _format_count_ratio(count, total):
    total = max(int(total), 1)
    return int(count), float(count) * 100.0 / float(total)


def _annotate_match_visual(vis_path, text_lines, logger=None):
    if not vis_path or not os.path.isfile(vis_path):
        return
    try:
        image = Image.open(vis_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        lines = [str(line) for line in text_lines if str(line).strip()]
        if not lines:
            return

        left = 8
        top = 8
        pad_x = 8
        pad_y = 6
        line_gap = 4
        text_width = 0
        text_heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = max(text_width, int(bbox[2] - bbox[0]))
            text_heights.append(int(bbox[3] - bbox[1]))
        box_w = text_width + 2 * pad_x
        box_h = sum(text_heights) + max(0, len(text_heights) - 1) * line_gap + 2 * pad_y
        draw.rectangle((left, top, left + box_w, top + box_h), fill=(0, 0, 0))

        cursor_y = top + pad_y
        for line, line_h in zip(lines, text_heights):
            draw.text((left + pad_x, cursor_y), line, fill=(255, 255, 0), font=font)
            cursor_y += line_h + line_gap

        image.save(vis_path)
    except Exception as exc:
        if logger is not None:
            logger.debug("匹配可视化标注失败: path=%s err=%s", vis_path, str(exc))


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
        match_mode="sparse",
        orientation_checkpoint="",
        orientation_mode="off",
        orientation_fusion_weight=0.5,
        orientation_topk=1,
        save_match_vis=False,
        match_vis_dir="",
        match_vis_max_save=200,
        logger=None,
        rotate=True,
        epoch=None):
    
    if logger is None:
        logger = logging.getLogger("game4loc.eval")
    logger.info("开始提取特征并计算相似度分数")
    logger.info("评估参数: ranks=%s, sdmk=%s, disk=%s, step_size=%s, with_match=%s, match_mode=%s", ranks_list, sdmk_list, disk_list, step_size, with_match, match_mode)
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
            logger=logger,
            match_mode=match_mode,
            sparse_save_final_vis=bool(save_match_vis),
            sparse_save_final_vis_dir=(match_vis_dir if str(match_vis_dir).strip() else None),
            sparse_save_final_vis_max=int(match_vis_max_save),
        )
        gallery_img_cache = OrderedDict()
        gallery_img_cache_size = 512
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
        logger.info("VOP 已加载: %s", orientation_checkpoint)
    elif with_match and match_mode == "sparse" and orientation_mode_key != "off":
        logger.warning("VOP 模式=%s 但未提供有效的 orientation_checkpoint，自动退化为常规 sparse 匹配。", orientation_mode_key)

    orientation_stats = {
        "count": 0,
        "time_sum": 0.0,
        "match_time_sum": 0.0,
        "hypothesis_sum": 0.0,
        "top_prob_sum": 0.0,
        "entropy_sum": 0.0,
        "concentration_sum": 0.0,
    }
    ma_thresholds = (3.0, 5.0, 10.0, 20.0)
    ma_hits = {threshold: 0 for threshold in ma_thresholds}
    final_match_stats = {
        "query_count": 0,
        "fallback_count": 0,
        "worse_than_coarse_count": 0,
        "identity_h_fallback_count": 0,
        "out_of_bounds_count": 0,
        "projection_invalid_count": 0,
        "retained_matches_sum": 0.0,
        "inliers_sum": 0.0,
        "inlier_ratio_sum": 0.0,
        "vop_forward_time_sum": 0.0,
        "matcher_time_sum": 0.0,
        "total_time_sum": 0.0,
    }

    ap = 0.0

    gallery_idx = {}
    for idx, gallery_img in enumerate(gallery_list):
        gallery_idx[gallery_img] = idx

    matches_list = []
    for query_i in query_list:
        pairs_list_i = pairs_dict[query_i]
        matches_i = []
        for pair in pairs_list_i:
            matches_i.append(gallery_idx[pair])
        matches_list.append(np.array(matches_i))

    matches_tensor = [torch.tensor(matches, dtype=torch.long) for matches in matches_list]

    query_num = img_features_query.shape[0]
    if logger is not None:
        logger.info("特征提取完成: 查询特征数量=%d, 图库图像数量=%d", query_num, len(gallery_list))
        if with_match and match_mode == "sparse":
            if query_yaw_list is None:
                logger.warning("sparse yaw 检查: query_yaw_list=None，将无法应用方向先验 (可能因 --use_yaw 未启用或数据未提供)")
            else:
                valid_yaws = []
                invalid_count = 0
                for y in query_yaw_list:
                    if y is None:
                        invalid_count += 1
                        continue
                    try:
                        yf = float(y)
                    except (TypeError, ValueError):
                        invalid_count += 1
                        continue
                    if not np.isfinite(yf):
                        invalid_count += 1
                        continue
                    valid_yaws.append(yf)

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

    t_metrics = time.perf_counter()
    progress_interval = max(1, query_num // 10)
    for i in tqdm(range(query_num), desc="Processing each query"):
        query_name = str(query_list[i])
        query_case_prefix = f"q{i+1:04d}_of_{query_num:04d}_{os.path.splitext(os.path.basename(query_name))[0]}"
        if logger is not None and with_match:
            logger.debug("=" * 100)
            logger.debug("Query[%d/%d] 开始: %s", i + 1, query_num, query_name)
        score = all_scores[i]    
        # predict index
        index = np.argsort(score)[::-1]
        top1_index = index[0]

        # with image match for finer loc
        match_loc = None
        match_info = None
        orientation_posterior = None
        hypotheses_evaluated = 0
        query_vop_time = 0.0
        query_match_time = 0.0
        if with_match:
            # 查询图每个 index 只用一次；图库 top1 在多查询中会复用，使用轻量 LRU 减少磁盘读图次数。
            if top1_index in gallery_img_cache:
                gallery_img = gallery_img_cache.pop(top1_index)
                gallery_img_cache[top1_index] = gallery_img
            else:
                gallery_img = gallery_loader.dataset[top1_index]
                gallery_img_cache[top1_index] = gallery_img
                if len(gallery_img_cache) > gallery_img_cache_size:
                    gallery_img_cache.popitem(last=False)
            query_img = query_loader.dataset[i]

        sparse_yaw0 = None
        sparse_yaw1 = None
        if with_match:
            if match_mode == "sparse":
                if query_yaw_list is not None and i < len(query_yaw_list) and query_yaw_list[i] is not None:
                    # D2S: satellite is treated as north-up (yaw=0), drone yaw comes from metadata.
                    sparse_yaw0 = 0.0
                    sparse_yaw1 = query_yaw_list[i]
            if use_orientation_model and orientation_mode_key == "prior_single":
                candidate_angles = getattr(orientation_model, "candidate_angles_deg", None) or [0.0]
                t_orientation = time.perf_counter()
                orientation_posterior = orientation_model.predict_posterior(
                    retrieval_model=model,
                    gallery_img=gallery_img,
                    query_img=query_img,
                    candidate_angles_deg=candidate_angles,
                    device=config.device,
                    gallery_branch="img2",
                    query_branch="img1",
                )
                query_vop_time = time.perf_counter() - t_orientation
                orientation_stats["time_sum"] += query_vop_time
                rotated_query_img = _rotate_query_tensor(query_img, orientation_posterior["top_angle_deg"])
                t_match = time.perf_counter()
                match_loc = matcher.est_center(
                    gallery_img,
                    rotated_query_img,
                    gallery_center_loc_xy_list[top1_index],
                    gallery_topleft_loc_xy_list[top1_index],
                    yaw0=None,
                    yaw1=None,
                    rotate=0.0,
                    case_name=f"{query_case_prefix}_vop_prior_single",
                )
                query_match_time = time.perf_counter() - t_match
                orientation_stats["match_time_sum"] += query_match_time
                hypotheses_evaluated = 1
            elif use_orientation_model and orientation_mode_key == "prior_topk":
                candidate_angles = getattr(orientation_model, "candidate_angles_deg", None) or [0.0]
                topk = max(1, min(int(orientation_topk), len(candidate_angles)))
                t_orientation = time.perf_counter()
                orientation_posterior = orientation_model.predict_posterior(
                    retrieval_model=model,
                    gallery_img=gallery_img,
                    query_img=query_img,
                    candidate_angles_deg=candidate_angles,
                    device=config.device,
                    gallery_branch="img2",
                    query_branch="img1",
                )
                query_vop_time = time.perf_counter() - t_orientation
                orientation_stats["time_sum"] += query_vop_time
                sorted_indices = list(np.argsort(np.asarray(orientation_posterior["probs"], dtype=np.float64))[::-1][:topk])
                best_match_loc = None
                best_match_info = None
                best_match_inliers = -1
                best_match_ratio = -1.0
                best_cand_angle = None
                total_topk_match_time = 0.0
                for rank_j, cand_index in enumerate(sorted_indices):
                    cand_angle = float(candidate_angles[int(cand_index)])
                    rotated_query_img = _rotate_query_tensor(query_img, cand_angle)
                    t_match = time.perf_counter()
                    candidate_match_loc = matcher.est_center(
                        gallery_img,
                        rotated_query_img,
                        gallery_center_loc_xy_list[top1_index],
                        gallery_topleft_loc_xy_list[top1_index],
                        yaw0=None,
                        yaw1=None,
                        rotate=0.0,
                        case_name=f"{query_case_prefix}_vop_prior_top{topk}_{rank_j}_{cand_angle:.1f}",
                        save_final_vis=False,
                    )
                    total_topk_match_time += time.perf_counter() - t_match
                    candidate_match_info = matcher.get_last_match_info() or {}
                    cand_kept = int(candidate_match_info.get("n_kept", 0))
                    cand_inliers = int(candidate_match_info.get("inliers", 0))
                    cand_ratio = float(cand_inliers) / float(max(cand_kept, 1))
                    if _prefer_match_candidate(cand_inliers, cand_ratio, best_match_inliers, best_match_ratio):
                        best_match_loc = candidate_match_loc
                        best_match_info = dict(candidate_match_info)
                        best_match_inliers = cand_inliers
                        best_match_ratio = cand_ratio
                        best_cand_angle = cand_angle
                if save_match_vis and best_cand_angle is not None:
                    rotated_query_img = _rotate_query_tensor(query_img, best_cand_angle)
                    match_loc = matcher.est_center(
                        gallery_img,
                        rotated_query_img,
                        gallery_center_loc_xy_list[top1_index],
                        gallery_topleft_loc_xy_list[top1_index],
                        yaw0=None,
                        yaw1=None,
                        rotate=0.0,
                        case_name=f"{query_case_prefix}_vop_prior_top{topk}_selected_{best_cand_angle:.1f}",
                        save_final_vis=True,
                    )
                    match_info = matcher.get_last_match_info() or {}
                else:
                    match_loc = best_match_loc
                    match_info = best_match_info
                query_match_time = total_topk_match_time
                orientation_stats["match_time_sum"] += query_match_time
                hypotheses_evaluated = topk
            else:
                t_match = time.perf_counter()
                match_loc = matcher.est_center(
                    gallery_img,
                    query_img,
                    gallery_center_loc_xy_list[top1_index],
                    gallery_topleft_loc_xy_list[top1_index],
                    yaw0=sparse_yaw0,
                    yaw1=sparse_yaw1,
                    rotate=rotate,
                    case_name=f"{query_case_prefix}_baseline_sparse",
                    save_final_vis=save_match_vis,
                )
                query_match_time = time.perf_counter() - t_match

            if match_info is None:
                match_info = matcher.get_last_match_info()

            if orientation_posterior is not None:
                orientation_stats["count"] += 1
                orientation_stats["top_prob_sum"] += float(orientation_posterior["top_prob"])
                orientation_stats["entropy_sum"] += float(orientation_posterior["entropy"])
                orientation_stats["concentration_sum"] += float(orientation_posterior["concentration"])
                orientation_stats["hypothesis_sum"] += float(max(hypotheses_evaluated, 1))

            dis_ori = get_dis_target(query_center_loc_xy_list[i], gallery_center_loc_xy_list[top1_index])
            is_fallback = bool(isinstance(match_info, dict) and match_info.get("fallback_to_center", False))
            if not is_fallback and match_loc is not None:
                is_fallback = (
                    float(match_loc[0]) == float(gallery_center_loc_xy_list[top1_index][0])
                    and float(match_loc[1]) == float(gallery_center_loc_xy_list[top1_index][1])
                )

            if is_fallback or match_loc is None:
                dis_match = dis_ori
                match_loc = gallery_center_loc_xy_list[top1_index]
            else:
                dis_match = get_dis_target(query_center_loc_xy_list[i], match_loc)

            match_info_dict = match_info if isinstance(match_info, dict) else {}
            vis_path = match_info_dict.get("final_vis_path")
            if vis_path:
                _annotate_match_visual(
                    vis_path,
                    [
                        f"Query {i + 1}/{query_num}",
                        f"Query: {os.path.basename(query_name)}",
                        f"Top1: {os.path.basename(str(gallery_list[top1_index]))}",
                        f"Coarse Dis: {dis_ori:.2f} m",
                        f"Matched Dis: {dis_match:.2f} m",
                        f"Fallback: {'yes' if is_fallback else 'no'}",
                    ],
                    logger=logger,
                )
            final_match_stats["query_count"] += 1
            final_match_stats["vop_forward_time_sum"] += float(query_vop_time)
            final_match_stats["matcher_time_sum"] += float(query_match_time)
            final_match_stats["total_time_sum"] += float(query_vop_time + query_match_time)
            final_match_stats["fallback_count"] += int(is_fallback)
            final_match_stats["worse_than_coarse_count"] += int(dis_match > (dis_ori + 1e-6))
            final_match_stats["identity_h_fallback_count"] += int(bool(match_info_dict.get("identity_h_fallback", False)))
            final_match_stats["out_of_bounds_count"] += int(bool(match_info_dict.get("out_of_bounds", False)))
            final_match_stats["projection_invalid_count"] += int(bool(match_info_dict.get("projection_invalid", False)))
            final_match_stats["retained_matches_sum"] += float(match_info_dict.get("n_kept", 0))
            final_match_stats["inliers_sum"] += float(match_info_dict.get("inliers", 0))
            final_match_stats["inlier_ratio_sum"] += float(match_info_dict.get("inlier_ratio", 0.0))
            dis_match_list.append(dis_match)
        else:
            dis_ori = get_dis_target(query_center_loc_xy_list[i], gallery_center_loc_xy_list[top1_index])
            dis_match_list.append(None)
        
        dis_ori_list.append(dis_ori)
        
        if logger is not None and with_match:
            fallback_str = " (回退粗检索)" if is_fallback else ""
            logger.debug(
                "Query[%d/%d] 结果: Query=%s | Top1Gallery=%s | 检索Dis=%.2fm | 匹配后Dis=%.2fm%s | retained=%s | inliers=%s | final_vis=%s",
                i + 1,
                query_num,
                query_name,
                gallery_list[top1_index],
                dis_ori,
                dis_match,
                fallback_str,
                str(match_info_dict.get("n_kept", "NA")),
                str(match_info_dict.get("inliers", "NA")),
                str(match_info_dict.get("final_vis_path", "")),
            )
            
        sdm_list.append(sdm(query_center_loc_xy_list[i], sdmk_list, index, gallery_center_loc_xy_list))

        current_dis = get_dis(query_center_loc_xy_list[i], index, gallery_center_loc_xy_list, disk_list, match_loc)
        dis_list.append(current_dis)
        final_top1_dis = float(current_dis[0])
        for threshold in ma_thresholds:
            if final_top1_dis < threshold:
                ma_hits[threshold] += 1

        top10_list.append(get_top10(index, gallery_list))
        loc1_x, loc1_y = gallery_center_loc_xy_list[index[0]]
        if with_match:
            loc1_x, loc1_y = match_loc
        loc1_list.append((query_center_loc_xy_list[i][0], query_center_loc_xy_list[i][1], loc1_x, loc1_y))

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
            logger.debug("评估进度: %d/%d (%.1f%%)", i + 1, query_num, (i + 1) * 100.0 / query_num)
    
    metrics_time = time.perf_counter() - t_metrics
    if with_match and matcher is not None:
        matcher.summarize_and_log()
    mAP = np.mean(all_ap)
    cmc = cmc / query_num

    sdm_list = np.mean(np.array(sdm_list), axis=0)
    dis_list = np.mean(np.array(dis_list), axis=0)

    # top 1%
    top1 = round(len(gallery_list)*0.01)

    string = []
    for r in ranks_list:
        string.append('Recall@{}: {:.4f}'.format(r, cmc[r-1] * 100))
    
    string.append('mAP: {:.4f}'.format(mAP * 100))

    for k in disk_list:
        string.append('SDM@{}: {:.4f}'.format(k, sum(sdm_list) / len(sdm_list) * 100))
    
    for i in range(len(disk_list)):
        string.append('Dis@{}: {:.4f}'.format(disk_list[i], dis_list[i]))

    result_str = ' - '.join(string)
    logger.info(result_str)
    logger.info(
        "MA@3m: %.4f - MA@5m: %.4f - MA@10m: %.4f - MA@20m: %.4f",
        float(ma_hits[3.0]) * 100.0 / float(max(query_num, 1)),
        float(ma_hits[5.0]) * 100.0 / float(max(query_num, 1)),
        float(ma_hits[10.0]) * 100.0 / float(max(query_num, 1)),
        float(ma_hits[20.0]) * 100.0 / float(max(query_num, 1)),
    )
    logger.info(
        "评估结果摘要: Recall@1=%.4f%%, Recall@5=%.4f%%, Recall@10=%.4f%%, mAP=%.4f%%",
        cmc[0] * 100,
        cmc[4] * 100 if len(cmc) > 4 else -1.0,
        cmc[9] * 100 if len(cmc) > 9 else -1.0,
        mAP * 100,
    )
    if with_match and final_match_stats["query_count"] > 0:
        final_query_count = int(final_match_stats["query_count"])
        fallback_count, fallback_ratio = _format_count_ratio(final_match_stats["fallback_count"], final_query_count)
        worse_count, worse_ratio = _format_count_ratio(final_match_stats["worse_than_coarse_count"], final_query_count)
        identity_count, identity_ratio = _format_count_ratio(final_match_stats["identity_h_fallback_count"], final_query_count)
        oob_count, oob_ratio = _format_count_ratio(final_match_stats["out_of_bounds_count"], final_query_count)
        invalid_count, invalid_ratio = _format_count_ratio(final_match_stats["projection_invalid_count"], final_query_count)
        logger.info(
            "稳健性统计(按查询汇总): fallback=%d(%.2f%%) worse-than-coarse=%d(%.2f%%) identity-H fallback=%d(%.2f%%) out-of-bounds=%d(%.2f%%) projection-invalid=%d(%.2f%%)",
            fallback_count,
            fallback_ratio,
            worse_count,
            worse_ratio,
            identity_count,
            identity_ratio,
            oob_count,
            oob_ratio,
            invalid_count,
            invalid_ratio,
        )
        logger.info(
            "最终匹配统计(按查询汇总): mean_retained_matches=%.2f mean_inliers=%.2f mean_inlier_ratio=%.4f queries=%d",
            float(final_match_stats["retained_matches_sum"]) / float(final_query_count),
            float(final_match_stats["inliers_sum"]) / float(final_query_count),
            float(final_match_stats["inlier_ratio_sum"]) / float(final_query_count),
            final_query_count,
        )
        logger.info(
            "细定位耗时(按查询汇总): mean_vop_forward_time=%.6fs/query mean_matcher_time=%.6fs/query mean_total_time=%.6fs/query",
            float(final_match_stats["vop_forward_time_sum"]) / float(final_query_count),
            float(final_match_stats["matcher_time_sum"]) / float(final_query_count),
            float(final_match_stats["total_time_sum"]) / float(final_query_count),
        )
    if orientation_stats["count"] > 0:
        orientation_count = max(int(orientation_stats["count"]), 1)
        logger.info(
            "VOP 摘要: count=%d mean_top_prob=%.4f mean_entropy=%.4f mean_concentration=%.4f mean_hypotheses=%.2f",
            int(orientation_stats["count"]),
            float(orientation_stats["top_prob_sum"]) / float(orientation_count),
            float(orientation_stats["entropy_sum"]) / float(orientation_count),
            float(orientation_stats["concentration_sum"]) / float(orientation_count),
            float(orientation_stats["hypothesis_sum"]) / float(orientation_count),
        )
        logger.info(
            "VOP 耗时: mean_posterior_time=%.6fs/query mean_match_time=%.6fs/query mean_total_time=%.6fs/query",
            float(orientation_stats["time_sum"]) / float(orientation_count),
            float(orientation_stats["match_time_sum"]) / float(orientation_count),
            float(orientation_stats["time_sum"] + orientation_stats["match_time_sum"]) / float(orientation_count),
        )
    logger.debug(
        "评估耗时统计 查询特征提取=%.6fs 图库推理=%.6fs 分数拼接=%.6fs 指标计算=%.6fs 查询数=%d 图库数=%d",
        query_extract_time,
        gallery_infer_time,
        score_concat_time,
        metrics_time,
        query_num,
        len(gallery_list),
    )
    
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

            if dis_match != None and dis_ori < dis_match:
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
