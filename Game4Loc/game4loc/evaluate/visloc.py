import torch
import numpy as np
from tqdm import tqdm
import gc
import time
from torch.cuda.amp import autocast
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from geopy.distance import geodesic

from ..matcher.gim_dkm import GimDKM


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
        sparse_phase1_min_inliers=10,
        sparse_angle_score_inlier_offset=None,
        sparse_use_multi_scale=True,
        sparse_save_final_vis=False,
        angle_experiment=False,
        wandb_run=None,
        epoch=None,
        logger=None,
    ):
    t_total = time.perf_counter()
    if logger is not None:
        logger.info("开始评估：提取特征并计算匹配分数")
        logger.debug(
            "评估参数：ranks=%s, sdmk=%s, disk=%s, step_size=%s, with_match=%s, match_mode=%s, rotate=%s, sparse_phase1_min_inliers=%s, sparse_angle_score_inlier_offset=%s, sparse_use_multi_scale=%s, sparse_save_final_vis=%s, angle_experiment=%s",
            ranks_list, sdmk_list, disk_list, step_size, with_match, match_mode, rotate, sparse_phase1_min_inliers, sparse_angle_score_inlier_offset, sparse_use_multi_scale, sparse_save_final_vis, angle_experiment,
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
            sparse_phase1_min_inliers=sparse_phase1_min_inliers,
            sparse_angle_score_inlier_offset=sparse_angle_score_inlier_offset,
            sparse_use_multi_scale=sparse_use_multi_scale,
            sparse_save_final_vis=sparse_save_final_vis,
        )

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
        sparse_yaw0 = None
        sparse_yaw1 = None
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
            match_loc_lon_lat = matcher.est_center(
                gallery_sample,
                query_sample,
                gallery_center_lon_lat,
                gallery_topleft_lon_lat,
                yaw0=sparse_yaw0,
                yaw1=sparse_yaw1,
                rotate=rotate,
                case_name=query_list[i],
            )
            if hasattr(matcher, "get_last_match_info"):
                match_info = matcher.get_last_match_info()
            else:
                match_info = getattr(matcher, "last_match_info", None)
            if hasattr(matcher, "get_last_angle_results"):
                angle_results = matcher.get_last_angle_results()
            match_loc_lat_lon = match_loc_lon_lat[1], match_loc_lon_lat[0]
            dis_match_list.append(get_dis_target(query_center_loc_xy_list[i], match_loc_lat_lon))
            match_loc = match_loc_lat_lon
        else:
            dis_match_list.append(None)
        
        dis_ori_list.append(get_dis_target(query_center_loc_xy_list[i], gallery_center_loc_xy_list[top1_index]))

        sdm_list.append(sdm(query_center_loc_xy_list[i], sdmk_list, index, gallery_center_loc_xy_list))

        dis_list.append(get_dis(query_center_loc_xy_list[i], index, gallery_center_loc_xy_list, disk_list, match_loc))
        if logger is not None and angle_experiment and with_match and match_mode == "sparse":
            logger.info(
                "[角度实验] Query=%s | Top1Gallery=%s | 原始Top1Dis=%.2fm | 最终Dis=%.2fm | 候选角度数=%d",
                query_list[i],
                gallery_list[top1_index],
                float(dis_ori_list[i]),
                float(dis_list[i][0]),
                len(angle_results),
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
        if epoch is not None:
            log_data["eval/epoch"] = int(epoch)
        wandb_run.log(log_data)
    
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
