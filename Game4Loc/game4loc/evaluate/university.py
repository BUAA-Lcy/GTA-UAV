import torch
import numpy as np
from tqdm import tqdm
import gc
import time
from ..trainer import predict


def evaluate(config,
                  model,
                  query_loader,
                  gallery_loader,
                  ranks=[1, 5, 10],
                  step_size=1000,
                  cleanup=True,
                  wandb_run=None,
                  epoch=None,
                  logger=None):
    t_total = time.perf_counter()
    if logger is not None:
        logger.info("开始评估：提取查询与图库特征")
        logger.debug("评估参数：ranks=%s, step_size=%s, cleanup=%s", ranks, step_size, cleanup)
    else:
        print("Extract Features:")

    t_extract = time.perf_counter()
    img_features_query, ids_query = predict(config, model, query_loader)
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader)
    extract_time = time.perf_counter() - t_extract
    
    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()
    
    if logger is not None:
        logger.info("开始计算相似度分数与检索指标")
    else:
        print("Compute Scores:")
    t_metrics = time.perf_counter()
    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0
    for i in range(len(ids_query)):
        ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i], img_features_gallery, gl)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    
    metrics_time = time.perf_counter() - t_metrics
    AP = ap/len(ids_query)*100
    
    CMC = CMC.float()
    CMC = CMC/len(ids_query) #average CMC
    
    # top 1%
    top1 = round(len(ids_gallery)*0.01)
    
    string = []
             
    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i-1]*100))
        
    string.append('Recall@top1: {:.4f}'.format(CMC[top1]*100))
    string.append('AP: {:.4f}'.format(AP))             
        
    result_str = ' - '.join(string)
    if logger is not None:
        logger.info(result_str)
        logger.info(
            "评估结果摘要：Recall@1=%.4f%%, Recall@5=%.4f%%, Recall@10=%.4f%%, AP=%.4f%%",
            CMC[0] * 100,
            CMC[4] * 100 if len(CMC) > 4 else -1.0,
            CMC[9] * 100 if len(CMC) > 9 else -1.0,
            AP,
        )
        logger.debug(
            "评估耗时统计：特征提取=%.6fs, 指标计算=%.6fs, 总耗时=%.6fs, 查询数=%d, 图库数=%d",
            extract_time,
            metrics_time,
            time.perf_counter() - t_total,
            len(ids_query),
            len(ids_gallery),
        )
    else:
        print(result_str)

    if wandb_run is not None:
        log_data = {
            "eval/recall@1": float(CMC[0] * 100),
            "eval/AP": float(AP),
            "time/eval_university/extract_s": extract_time,
            "time/eval_university/metrics_s": metrics_time,
            "time/eval_university/total_s": time.perf_counter() - t_total,
            "eval/query_num": int(len(ids_query)),
            "eval/gallery_num": int(len(ids_gallery)),
        }
        if len(CMC) > 4:
            log_data["eval/recall@5"] = float(CMC[4] * 100)
        if len(CMC) > 9:
            log_data["eval/recall@10"] = float(CMC[9] * 100)
        if epoch is not None:
            log_data["eval/epoch"] = int(epoch)
        wandb_run.log(log_data)
    
    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()
        #torch.cuda.empty_cache()
    
    return CMC[0]


def eval_query(qf,ql,gf,gl):

    score = gf @ qf.unsqueeze(-1)
    
    score = score.squeeze().cpu().numpy()
 
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]    

    # good index
    query_index = np.argwhere(gl==ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl==-1)
    
    
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc




