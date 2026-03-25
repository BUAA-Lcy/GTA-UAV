import torch
import numpy as np
from tqdm import tqdm
import gc
import time
from collections import OrderedDict
from torch.cuda.amp import autocast
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from PIL import Image
import pickle
import os

from ..matcher.gim_dkm import GimDKM


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
        logger=None,
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
        matcher = GimDKM(device=config.device, logger=logger, match_mode=match_mode)
        gallery_img_cache = OrderedDict()
        gallery_img_cache_size = 512

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
                logger.warning("sparse yaw 检查: query_yaw_list=None，将无法应用方向先验 (可能因 --ignore_yaw 被禁用或数据未提供)")
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
        score = all_scores[i]    
        # predict index
        index = np.argsort(score)[::-1]
        top1_index = index[0]

        # with image match for finer loc
        match_loc = None
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

            sparse_yaw0 = None
            sparse_yaw1 = None
            if match_mode == "sparse":
                if query_yaw_list is not None and i < len(query_yaw_list) and query_yaw_list[i] is not None:
                    # D2S: satellite is treated as north-up (yaw=0), drone yaw comes from metadata.
                    sparse_yaw0 = 0.0
                    sparse_yaw1 = query_yaw_list[i]
            match_loc = matcher.est_center(
                gallery_img,
                query_loader.dataset[i],
                gallery_center_loc_xy_list[top1_index],
                gallery_topleft_loc_xy_list[top1_index],
                yaw0=sparse_yaw0,
                yaw1=sparse_yaw1,
            )
            dis_match_list.append(get_dis_target(query_center_loc_xy_list[i], match_loc))
        else:
            dis_match_list.append(None)
        
        dis_ori_list.append(get_dis_target(query_center_loc_xy_list[i], gallery_center_loc_xy_list[top1_index]))
            
        sdm_list.append(sdm(query_center_loc_xy_list[i], sdmk_list, index, gallery_center_loc_xy_list))

        dis_list.append(get_dis(query_center_loc_xy_list[i], index, gallery_center_loc_xy_list, disk_list, match_loc))

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
        "评估结果摘要: Recall@1=%.4f%%, Recall@5=%.4f%%, Recall@10=%.4f%%, mAP=%.4f%%",
        cmc[0] * 100,
        cmc[4] * 100 if len(cmc) > 4 else -1.0,
        cmc[9] * 100 if len(cmc) > 9 else -1.0,
        mAP * 100,
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
