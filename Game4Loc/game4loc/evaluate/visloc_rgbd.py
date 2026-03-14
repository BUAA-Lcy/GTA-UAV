import torch
import numpy as np
from tqdm import tqdm
import gc
import time
from torch.cuda.amp import autocast
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from geopy.distance import geodesic


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


def get_dis(query_loc, index, gallery_loc_xy_list, disk_list):
    query_lat, query_lon = query_loc
    dis_list = []
    for k in disk_list:
        dis_sum = 0.0
        for i in range(k):
            idx = index[i]
            gallery_lat, gallery_lon = gallery_loc_xy_list[idx]
            dis = geodesic((query_lat, query_lon), (gallery_lat, gallery_lon)).meters
            dis_sum += dis
        dis_list.append(dis_sum / k)

    return dis_list


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


def evaluate(config,
                model,
                query_loader,
                gallery_loader,
                query_list,
                query_loc_xy_list,
                gallery_list,
                gallery_loc_xy_list,
                pairs_dict,
                ranks_list=[1, 5, 10],
                sdmk_list=[1, 3, 5],
                disk_list=[1, 3, 5],
                step_size=1000,
                cleanup=True,
                wandb_run=None,
                epoch=None,
                logger=None):
    t_total = time.perf_counter()
    if logger is not None:
        logger.info("开始评估：提取特征并计算匹配分数")
        logger.debug("评估参数：ranks=%s, sdmk=%s, disk=%s, step_size=%s", ranks_list, sdmk_list, disk_list, step_size)
    else:
        print("Extract Features and Compute Scores:")

    t_query = time.perf_counter()
    img_features_query = predict(config, model, query_loader)
    query_extract_time = time.perf_counter() - t_query

    all_scores = []
    model.eval()
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

    all_ap = []
    cmc = np.zeros(len(gallery_list))
    sdm_list = []
    dis_list = []

    t_metrics = time.perf_counter()
    for i in range(query_num):
        str_i = query_list[i].split('_')[0]
        score = all_scores[i] * gallery_mapi_idx[str_i]
        index = np.argsort(score)[::-1]

        sdm_list.append(sdm(query_loc_xy_list[i], sdmk_list, index, gallery_loc_xy_list))
        dis_list.append(get_dis(query_loc_xy_list[i], index, gallery_loc_xy_list, disk_list))

        good_index_i = np.isin(index, matches_tensor[i])

        y_true = good_index_i.astype(int)
        y_scores = np.arange(len(y_true), 0, -1)
        if np.sum(y_true) > 0:
            ap = average_precision_score(y_true, y_scores)
            all_ap.append(ap)

        match_rank = np.where(good_index_i == 1)[0]
        if len(match_rank) > 0:
            cmc[match_rank[0]:] += 1
    metrics_time = time.perf_counter() - t_metrics

    mAP = np.mean(all_ap)
    cmc = cmc / query_num
    sdm_list = np.mean(np.array(sdm_list), axis=0)
    dis_list = np.mean(np.array(dis_list), axis=0)

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
            "time/eval_visloc_rgbd/query_extract_s": query_extract_time,
            "time/eval_visloc_rgbd/gallery_infer_s": gallery_infer_time,
            "time/eval_visloc_rgbd/score_concat_s": score_concat_time,
            "time/eval_visloc_rgbd/metrics_s": metrics_time,
            "time/eval_visloc_rgbd/total_s": time.perf_counter() - t_total,
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

    if cleanup:
        del img_features_query, gallery_features_batch, scores_batch
        gc.collect()

    return cmc[0]
