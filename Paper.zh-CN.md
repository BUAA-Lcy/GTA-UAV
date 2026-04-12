# Paper.md

本文档是当前论文草稿的写作简报。

它不是实验日志，而是一份面向论文稿件的总结，说明：

- 这篇论文真正要写什么；
- 方法应该如何表述；
- 哪些实验结果已经足够扎实、可以写进正文；
- 哪些结果需要降级、弱化，或作为局限性保留。


# 1. 当前论文定位

这篇论文的主题是：

> **检索之后的精细定位（post-retrieval fine localization），用于无人机到卫星图像地理定位**

在论文叙事中，检索阶段应保持固定。方法贡献只从 retrieval top-1 已经选定之后开始。

对于当前草稿，写作优先级应为：



1. **GTA-UAV same-area**
2. **UAV-VisLoc 03/04 same-area 紧凑协议**

这和单纯按 benchmark 排名来组织不同，原因很简单：

- GTA same-area 提供了更大规模的稀疏侧验证；
- `03/04` 目前提供了清晰、容易写出的 VisLoc 故事线；

因此对于当前论文草稿：

- GTA 应承载 **更大规模 same-area 的稀疏比较**；承载 **主叙事**；

- `03/04` 应承载次级迁移检查 以及在真实场景下数据集的验证

  


# 2. 需要写出的核心论点

论文应围绕以下主张展开：

> 检索之后的精细定位失败，主要原因并不是“匹配点太少”，而是朝向歧义、不可靠对应关系以及不稳定几何三者共同作用。视觉朝向后验（VOP）能够把朝向不确定性转化为少量有用的角度假设，从而在不付出稠密匹配代价的前提下，显著增强稀疏几何验证。

当前较稳妥的表述版本是：

> 在当前仓库中，`VOP + sparse` 是最强的稀疏精细定位路线。它能稳定优于显式稀疏基线，同时保留相对稠密匹配的大幅运行时间优势；但在当前 GTA same-area 与 UAV-VisLoc `03/04` 的正式对比中，dense 仍然在绝对精度与稳健性上更强。

> `VOP + sparse` 已经在所有数据集上同时大幅度取得速度优势，并取得了接近dense的精度。

当前证据并不支持这一说法。


# 3. 为什么这篇论文有意思

问题诊断需要写得非常明确：

1. retrieval top-1 本身已经可能是正确的；
2. 但定位仍会失败，因为无人机图像相对北向朝上的卫星瓦片存在朝向歧义；
3. 朝向歧义会破坏对应关系质量；
4. 一旦对应关系不稳定，几何也会跟着不稳定。

论文里要保留的关键句子是：

> 瓶颈并不只是匹配数太少。真正的瓶颈是朝向歧义、不可靠对应关系与不稳定几何之间的相互作用。

这个诊断很重要，因为它解释了为什么：

- 朴素稀疏匹配过于脆弱；
- 暴力旋转搜索既昂贵又不优雅；
- VOP 是一种有原则的 proposal 机制，而不仅仅是另一个启发式技巧。


# 4. 方法概述

方法应被描述为一个检索后的流水线：

1. 保持 retrieval 固定；
2. 用 VOP 预测离散候选角度上的后验分布；
3. 保留 top-`k` 个有用角度假设；
4. 只在这些假设上运行稀疏匹配与几何验证；
5. 根据几何质量选择最终结果。

推荐术语：

- **visual orientation posterior (VOP)**
- **useful-angle proposer**
- **top-k angle hypotheses**
- **geometry-guided verification**

避免把 VOP 描述成：

- 精确角度回归器；
- 几何的替代品；
- 依赖元数据的 yaw 模块。

最合适的表述是：

> 一个轻量模块，将稀疏匹配预算分配给少量几何上更有希望的朝向。


# 5. VOP 机制与训练

这是论文中最具方法特色的部分，需要解释清楚。

## 5.1 VOP 预测什么

VOP 预测的是离散角度集合上的后验：

- `Theta = {theta_1, ..., theta_M}`

它**不是**预测单个精确的连续角度。

这很重要，因为观察到的角度误差曲面往往：

- 是多峰的；
- 更像宽峰而不是尖锐单峰；
- 在多个可接受朝向附近局部较平坦。

因此正确的建模对象是：

- 不是“唯一真实角度”；
- 而是一个 **有用角度集合**。

## 5.2 VOP 架构

VOP 工作在冻结的 retrieval 特征图之上。

对于每个候选角度：

1. 旋转 query 特征图；
2. 与 gallery 特征图结合；
3. 用一个轻量头输出该角度的 logit；
4. 在所有角度上做 softmax。

论文应强调两个性质：

- 相对稠密匹配，它是轻量的；
- 它是在稀疏几何**之前**处理朝向，而不是改 matcher 内部实现。

## 5.3 Teacher 构建

Teacher target 的构建方式是：对每个训练对在一组离散候选角度下进行评估，并记录几何之后的最终定位质量。

对于每个 pair 和角度，teacher 会保存：

- 最终定位距离；
- 最优角度；
- 最优距离；
- 次优距离；
- 距离间隔；
- 一个派生出的 soft target profile。

这里写作上最重要的一点是：

> 监督信号建立在最终定位结果之上，而不是原始位姿元数据之上。

## 5.4 有用角度集合监督

当前主线目标为：

- `U_i = {theta | d_i(theta) <= d_i* + delta}`

其中：

- `delta = 5m`

这表示模型应恢复所有对定位来说“足够好”的角度，而不是只恢复 teacher argmax 碰巧对应的那一个角度。

## 5.5 Pair 置信度加权

当前默认还使用 pair weighting：

- `w_i = 1 / (1 + exp((d_i* - c) / s))`

其中：

- `c = 30m`
- `s = 10m`

这样做会降低噪声大、质量弱的 teacher pair 权重，而不是直接把它们丢掉。

## 5.6 当前默认配方

当前应写入论文的方法配方为：

- `supervision_mode = useful_bce`
- `useful_delta_m = 5`
- `ce_weight = 1.0`
- `pair_weight_mode = best_distance_sigmoid`
- `pair_weight_center_m = 30`
- `pair_weight_scale_m = 10`
- `orientation_topk = 4`

这就是当前应作为主要贡献写出的那条方法线。


# 6. 推理时真正比较的是什么

论文必须精确定义被比较的流水线。

## 6.1 Dense DKM

1. retrieve top-1；
2. 运行稠密对应估计；
3. 采样稠密对应；
4. 估计几何；
5. 投影 gallery center。

## 6.2 Sparse

1. retrieve top-1；
2. 用默认 multi-scale 运行稀疏匹配；
3. 估计几何；
4. 投影 gallery center。

## 6.3 Rotate Baseline

rotate baseline 必须被准确定义为：

- 稀疏匹配；
- 四个候选旋转；
- 最终假设选择规则为：
  - inlier 数最多；
  - 如并列，则按 inlier ratio 打破。

不要用模糊表述如 “sparse + rotate”。

## 6.4 Ours

1. retrieve top-1；
2. 计算 VOP posterior；
3. 保留 top-`k=4` 个角度假设；
4. 在这些假设上运行稀疏匹配；
5. 由几何选择最终结果。

# 7. 当前草稿的 benchmark 定位

## 7.1 将 GTA Same-Area 作为主验证集合

GTA same-area 有价值，因为：

- 它规模更大；
- 它在更接近真实规模的条件下确认了稀疏侧排序；
- 它为 `VOP > sparse` 和 `VOP > rotate` 提供了更强的 same-area 论据。

现在匹配设置下的完整 dense 行也已经完成，这使 GTA 可以用于在 matched retrieval 下明确比较 `ours-versus-dense`。

但结果依然不是一个干净的 `ours > dense` 故事：

- `dense` 稳健性上仍然更强；
- `sparse + VOP` 仍然显著更快，并且依旧是最强的稀疏路线。
- `sparse + VOP` 已经接近了dense级别的精度，达到了同一级别的水准

## 7.2 将 VisLoc 作为在真实数据集下的验证

- 它是目前唯一一个稀疏侧故事相对足够强的 VisLoc 划分；
- 它展示了清晰的方法差异；
- 它提供了较易讲述的速度与精度权衡故事；
- 也是 VOP 监督分析最成熟的划分。

重要协议事实：

- split 文件：
  - `data/UAV_VisLoc_dataset/same-area-drone2sate-test.json`
- evaluation mode：
  - `test_mode=pos`
- json 中 query 总数：
  - `302`
- 实际参与精细定位评估的 query：
  - `116`
- gallery 大小：
  - `17528`

这个紧凑协议上的通用 retrieval 指标为：

- `Recall@1 = 92.2414`
- `Recall@5 = 99.1379`
- `Recall@10 = 100.0000`
- `AP = 95.6691`

现在写作时，主 VisLoc 表格应使用这套协议。


# 8. 可直接写入论文的实验总结

这一节对起草最重要。

## 8.1 GTA Same-Area：大规模的稀疏侧确认

当前写作应使用刷新后的完整 matched main-table 行。四条核心方法行之外，可在同一张表里追加一条补充外部 matcher 行 `LoFTR`：

| 变体                | Recall@1 | Recall@5 | Recall@10 |   mAP |  Dis@1 | MA@20 | fallback | worse-than-coarse | mean total time |
| ------------------- | -------: | -------: | --------: | ----: | -----: | ----: | -------: | ----------------: | --------------: |
| dense DKM           |    91.11 |    99.39 |     99.54 | 94.81 |  50.11 | 54.81 |    1.57% |            12.00% |         4.0410s |
| LoFTR               |    91.11 |    99.39 |     99.54 | 94.81 | 130.66 | 16.93 |    4.01% |            54.98% |         0.6972s |
| sparse              |    91.11 |    99.39 |     99.54 | 94.81 | 108.47 | 14.61 |   58.41% |            18.47% |         0.0651s |
| sparse + rotate90   |    91.11 |    99.39 |     99.54 | 94.81 |  77.50 | 36.45 |   11.97% |            21.49% |         0.2831s |
| sparse + VOP (ours) |    91.11 |    99.39 |     99.54 | 94.81 |  52.59 | 43.54 |   12.02% |            13.30% |         0.3044s |
|                     |          |          |           |       |        |       |          |                   |                 |

关键 GTA 对比：

### Ours vs sparse

- `Dis@1`: `108.47 -> 52.59` (`-55.88m`)
- `MA@20`: `14.61% -> 43.54%` (`+28.93pp`)

### Ours vs rotate

- `Dis@1`: `77.50 -> 52.59` (`-54.91m`)
- `MA@20`: `36.45% -> 43.54%` (`+7.09pp`)
- `worse-than-coarse`: `21.49% -> 13.30%` (`-8.19pp`)
- runtime: `0.2831s -> 0.3044s` (`+0.0213s`)

### Ours vs LoFTR

- `Dis@1`: `130.66 -> 52.59` (`-78.07m`)
- `MA@20`: `16.93% -> 43.54%` (`+26.61pp`)
- `fallback`: `4.01% -> 12.02%`（LoFTR 更少回退，但最终定位质量明显更差）
- runtime: `0.6972s -> 0.3044s`（ours 约快 `2.29x`）

这是一个很有价值的支撑结果，因为它在更大规模的 same-area benchmark 上确认了：

- rotate baseline 是必要的；
- `VOP + sparse` 仍优于这一显式 rotate baseline；
- 相比 rotate，额外运行时间很小。

### Ours vs dense

- `dense` 在绝对质量上更强：
- 但 `dense` 慢得多：
  - `4.0410s/query` vs `0.3044s/query`
  - 大约慢 `13.3x`

所以 GTA 现在适合用于支持：

- `VOP > sparse`
- `VOP > rotate`
- matched 的 `ours-versus-dense` 比较

但正确解读仍是：

- **ours 是最强的稀疏行，而且快得多，精度上达到了和dense的同级别水准**

## 8.2 UAV-VisLoc `03/04`：方法级主表

这里不应再把 `current teacher / clean30 / top-k` 这类 supervision 变体直接放进主比较表。`03/04` 的正文主表应该按方法级别组织为：

- `dense-DKM`
- `LoFTR`
- `SuperPoint`
- `SuperPoint + Rotate`
- `ours`

推荐主表如下：

| 方法 | Dis@1 | MA@3 | MA@5 | MA@10 | MA@20 | fallback | worse-than-coarse | mean inliers | total time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense DKM, no rotate | 32.45 | 2.59 | 4.31 | 13.79 | 49.14 | 1.72% | 25.86% | 4202.6 | 11.5241s |
| LoFTR | 86.57 | 1.72 | 2.59 | 5.17 | 18.97 | 2.59% | 62.07% | 49.8 | 0.8627s |
| SuperPoint | 76.77 | 0.86 | 0.86 | 0.86 | 6.90 | 50.00% | 81.03% | 22.6 | 0.0943s |
| SuperPoint + Rotate | 59.47 | 0.00 | 0.00 | 1.72 | 24.14 | 17.24% | 53.45% | 37.2 | 0.2942s |
| Ours (SuperPoint + VOP) | 38.86 | 0.86 | 2.59 | 8.62 | 37.93 | 11.21% | 40.52% | 43.2 | 0.3408s |

这张表应这样解读：

- `dense DKM` 仍然是 `03/04` 上绝对表现最强的一行；

- `LoFTR` 的回退率很低，但最终 `Dis@1` 明显偏高，说明它在这个协议上并没有转化成更好的最终定位质量；

- `SuperPoint` 裸稀疏匹配很弱，说明这个问题不能只靠“直接跑一次稀疏几何”解决；

- `SuperPoint + Rotate` 明显提升，说明方向歧义确实是核心瓶颈；

- `Ours` 进一步稳定优于显式 rotate baseline，说明 VOP 提供的不是简单暴力搜索替代，而是更有效的有用角度提议机制；

- 与 `dense` 相比，`ours` 仍然更弱，但运行时间显著更低；

  




## 8.4 Paper7：次级的负面 / 混合证据

Paper7 应简要总结，而不是放到前台。

| 变体 | Dis@1 | MA@20 | fallback | worse-than-coarse | mean total time |
|---|---:|---:|---:|---:|---:|
| dense DKM | 241.40 | 35.51 | 8.09% | 48.04% | 4.1128s |
| sparse | 274.68 | 10.44 | 43.60% | 78.59% | 0.0747s |
| sparse + rotate90 + inlier-count | 258.18 | 18.28 | 16.97% | 62.40% | 0.3111s |
| sparse + VOP | 257.94 | 25.59 | 15.93% | 60.57% | 0.3291s |

正确解读是：

- `VOP + sparse` 仍然是最强的 sparse 行；
- 但所有 sparse 行的绝对表现都偏弱；
- dense 在 headline accuracy 上仍更强；
- 因此在当前草稿中，不应把 Paper7 当作主要的正向视觉 benchmark。

如果 Paper7 出现在正文中，也应表述为：

> 这是一个更困难的迁移场景，其中稀疏侧排序仍被保留，但与 dense 的绝对性能差距依然存在。


# 9. 论文中应该怎么写

## 9.1 推荐 headline result

对于当前草稿，最干净的 headline 是：

> 在 GTA same-area 上，VOP 引导的稀疏匹配显著优于普通 sparse、显式的 rotate90 稀疏基线以及补充外部 matcher LoFTR，同时保留了相对于 dense DKM 的数量级速度优势；但 dense DKM 在绝对精度与稳健性上仍然更强。在紧凑的 UAV-VisLoc `03/04` 协议上，方法级比较也呈现相同排序：`Ours` 明显优于原始 `SuperPoint` 与 `SuperPoint + Rotate`，达到了和dense同一级别的精度水平。

## 9.2 推荐的方法主张

方法应被表述为：

- 一个有原则的朝向不确定性模块；
- 一种保持稀疏效率优势的精细定位策略；
- 当前实验中最强的稀疏路线。

尽量**不要**把它表述成：

- 通用的 dense 替代方案；
- 已经在所有地方都比 dense 更好。


# 10. 草稿应包含哪些章节

## 10.1 引言

引言应说明：

1. retrieval 成功并不等于 post-retrieval localization 被解决；
2. dense matching 强但昂贵；
3. naive sparse matching 便宜但脆弱；
4. 缺失的关键是在稀疏几何之前处理朝向不确定性的机制；
5. VOP 提供了这个机制。

## 10.2 方法

推荐的方法小节：

1. 问题设定
2. 失败诊断
3. VOP 形式化
4. teacher 构建
5. useful-angle set supervision
6. pair-confidence weighting
7. top-`k` 稀疏验证

## 10.3 实验

推荐的实验小节：

1. 配套 benchmark：GTA same-area 
2. VisLoc benchmark：
3. 监督比较
5. 运行时间与局限性


# 11. 需要准备的表格与图

## 11.1 主图

一张方法图应展示：

1. retrieval top-1 tile
2. VOP 在角度上的 posterior
3. top-`k` angle proposals
4. sparse matching 与 geometry verification
5. 最终定位输出



## 11.2 GTA 主表

使用一张紧凑的支撑表：

1. dense DKM

2. LoFTR

3. SP+LG

4. SP+LG+rotate90

5. Ours

   

## 11.2 UAV-VisLoc表

当前草稿的主表应使用 `03/04` 紧凑协议。

1. dense DKM
2. LoFTR
3. SP+LG
4. SP+LG+rotate90
5. Ours




# 12. 论断纪律

## 12.1 可以安全提出的论断

论文可以安全声称：

- post-retrieval fine localization 是一个独立于 retrieval 的问题；
- 朝向歧义是精细定位失败的核心原因之一；
- VOP 是关于有用角度假设的、有原则的离散后验；
- useful-angle set supervision 比 single-angle supervision 更符合问题结构；
- `VOP + sparse` 是当前实验中最强的稀疏路线；
- `VOP + sparse` 保留了相对于 dense matching 的显著运行时间优势；
- 在 GTA same-area 上，把 `LoFTR` 作为补充外部基线加入主表是合理的；
- 但当前正式协议下，`dense DKM` 仍然在绝对精度与稳健性上强于 `VOP + sparse`。
