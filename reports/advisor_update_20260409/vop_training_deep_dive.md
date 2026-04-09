# VOP 训练过程深度说明

## 文档目的

这份文档专门回答一个问题：

> 当前仓库里的 `VOP` 到底是怎么训练出来的？

这里不只讲“高层思路”，而是尽量把代码里的真实实现拆开说明，包括：

- 训练样本是怎么构造的；
- teacher 是怎么从定位结果里反推出来的；
- 模型到底吃什么、吐什么；
- 两种监督模式分别怎么定义；
- 每个 loss 在代码里具体起什么作用；
- 当前 checkpoint 的实际配置是什么；
- 以及一些容易被忽略、但会影响理解的实现细节。

---

## 1. 一句话总述

当前 VOP 的训练，不是端到端地直接优化最终定位误差，也不是重新训练 retrieval backbone。  
它更像一个 **挂在检索特征之上的轻量方向头**：

1. 先离线生成 teacher cache；
2. teacher cache 里保存每个 query-gallery 对在 `36` 个角度上的定位误差；
3. 再把这条角度-误差曲线转成监督信号；
4. 用这个监督去训练一个小型方向后验头；
5. 推理时再让这个头给出最可能的 `top-k` 个角度。

换句话说：

- **teacher** 来自“真实跑出来的定位结果”；
- **student** 是一个轻量方向分类 / 排序头；
- **backbone** 主要还是原来的检索模型；
- **目标** 是让模型学会：给定 query 和 gallery，哪些角度更值得进入后续细定位。

---

## 2. 当前训练链路里有哪些脚本

和训练直接相关的核心文件有：

- [train_vop.py](/home/lcy/Workplace/GTA-UAV/Game4Loc/train_vop.py)
- [build_vop_teacher.py](/home/lcy/Workplace/GTA-UAV/Game4Loc/build_vop_teacher.py)
- [vop.py](/home/lcy/Workplace/GTA-UAV/Game4Loc/game4loc/orientation/vop.py)

辅助分析文件有：

- [analyze_vop.py](/home/lcy/Workplace/GTA-UAV/Game4Loc/analyze_vop.py)

从流程上看，可以拆成三步：

1. `build_vop_teacher.py`  
   先构造 teacher cache。
2. `train_vop.py`  
   再用 teacher cache 训练方向头。
3. `analyze_vop.py`  
   最后做可分性分析、top-k 诊断和推理阶段统计。

---

## 3. 第一步：teacher cache 是怎么做出来的

这一部分非常关键，因为当前 VOP 的监督信号不是人工标注角度，而是由后续定位器“反推”出来的。

### 3.1 teacher 的基本输入

`build_vop_teacher.py` 读入的是训练对元数据文件，例如当前主线里用到的：

- `same-area-drone2sate-train.json`

teacher cache 当前主版本的元数据是：

- `rotate_step = 10.0`
- `temperature_m = 25.0`
- `img_size = 384`
- `model = vit_base_patch16_rope_reg1_gap_256.sbb_in1k`
- `checkpoint_start = ./pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth`
- `num_records = 502`

对应缓存文件是：

- `/home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/teacher_0407_full.pt`

### 3.2 每个样本到底是什么

teacher 构造时，会遍历 `pairs_meta_file` 里的每个 query 记录，但只保留那些至少有一个正样本卫星图的 query。

对于每条记录，它只取：

- `drone_img_name` 作为 query
- `pair_pos_sate_img_list[0]` 作为当前 teacher 对应的 gallery

也就是说，**当前 teacher 不是对所有正样本 tile 做监督，而是只取正样本列表中的第一个卫星 tile。**

这一点很重要：

- 当前 VOP 训练的对象，是一个固定的 `query -> top-1 positive tile` 对；
- 它不是在训练一个“对多个正样本都稳”的角度头。

### 3.3 角度离散化方式

候选角度由 `build_rotation_angle_list(10.0)` 生成。

当前一共是 `36` 个角度：

```text
[0, 10, 20, ..., 170, -180, -170, ..., -10]
```

也就是说：

- 整个圆周被离散成 36 个 bin；
- 顺序不是 `-180 -> 180`，而是先正向到 `170`，再接 `-180 -> -10`；
- 这个顺序在 teacher、train、inference 三边必须一致。

代码里也专门做了这个一致性检查：

- 如果 teacher 里的角度列表和 `rotate_step` 推导出的列表不一致，训练会直接报错。

### 3.4 单个 teacher 样本是怎么算出来的

对每个 query-gallery 对，teacher 构造逻辑是：

1. 读入 query 图和 gallery 图；
2. 由 gallery tile 名恢复卫星 tile 的中心经纬度和左上角经纬度；
3. 调 matcher：

```python
matcher.est_center(..., rotate=args.rotate_step)
```

这里本质上是在跑一个按 `10°` 步长的旋转搜索。

4. 取回 `matcher.get_last_angle_results()`；
5. 只保留 `phase == 1` 的角度结果；
6. 按 `search_angle` 排序；
7. 对每个角度结果：
   - 用单应矩阵把匹配中心投影回地理位置；
   - 如果投影失败，就记成 `inf`；
   - 如果成功，就用 `geodesic` 计算预测位置与 query 真值之间的米级误差；
8. 得到一个长度为 `36` 的 `distances_m` 列表。

换句话说，teacher 不是直接来自“图像看起来像哪个角度”，而是来自：

> 如果我真的在这个角度上跑完匹配和投影，最终定位误差是多少？

这就是当前 teacher 最有价值也最昂贵的地方。

### 3.5 soft teacher 是怎么从误差变成分布的

teacher 并不是 one-hot 角度标签，而是一个软分布。

构造方式是：

```text
logit_i = - distance_i / temperature_m
prob_i = softmax(logit_i)
```

其中：

- `distance_i` 是第 `i` 个角度下的最终定位误差；
- `temperature_m` 当前默认是 `25m`；
- 无效角度的 logit 会被设成一个很小的常数 `-40`。

直观上理解：

- 定位误差越小，teacher 概率越高；
- 误差差距不大的多个角度，会一起拿到较高概率；
- 所以 teacher 天然就允许“多角度都合理”的情况存在。

### 3.6 teacher cache 里保存了什么

每条 teacher record 里目前至少有这些字段：

- `query_name`
- `gallery_name`
- `query_path`
- `gallery_path`
- `query_loc_lat_lon`
- `candidate_angles_deg`
- `distances_m`
- `target_probs`
- `best_angle_deg`
- `best_index`
- `best_distance_m`
- `second_distance_m`
- `distance_gap_m`

这意味着训练阶段其实并不需要再跑 matcher；
它只需要读入：

- 图像路径；
- 36 个角度上的误差；
- 和由误差导出的概率分布。

### 3.7 当前 teacher 的统计特征

当前主 teacher cache 的 summary 是：

- 样本数：`502`
- 平均熵：`0.7484`
- 平均最优概率：`0.1533`
- 平均 top1-top2 概率差：`0.0348`
- 平均最优距离：`16.09m`
- 平均 top1-top2 距离差：`5.27m`
- `distance_gap >= 5m` 的样本数：`161`

这些数字非常重要，它们说明：

- teacher 大多数时候并不尖锐；
- 最优角通常不是“一家独大”；
- 但又不是完全不可分，因为还有一部分样本的 top1 / top2 差距已经不小。

这也是为什么后面训练里会同时出现：

- 软分布对齐；
- 排序约束；
- 和部分样本上的硬分类约束。

---

## 4. 第二步：训练数据集在干什么

teacher cache 建好之后，`train_vop.py` 会把它包成 `VOPDataset`。

### 4.1 正样本行

正常情况下，一个样本会返回：

- `query_img`
- `gallery_img`
- `target_probs`
- `distances`
- `best_index`
- `distance_gap`
- `target_entropy`

这些值都是从 teacher cache 直接读出来的。

### 4.2 负样本是怎么造出来的

训练里有一个很容易被忽略、但非常重要的细节：

- `neg_prob` 默认是 `0.25`

这意味着每次取样时，有 `25%` 概率会把当前 query 和一个随机抽到的**错误 gallery** 配对。

在这种负样本情况下，dataset 会做这些事情：

- `gallery_path` 换成别的记录的 gallery；
- `target_probs` 设成均匀分布；
- `distances` 全部设成 `inf`；
- `best_index = -1`
- `distance_gap = inf`
- `target_entropy = 1.0`

也就是说，负样本在训练里的语义是：

> 这对 query-gallery 根本不匹配，所以模型不应该偏向某个特定角度，而应该输出尽量平的分布。

### 4.3 验证集不会做负样本注入

这也是一个实现细节：

- `train_dataset` 用 `neg_prob = args.neg_prob`
- `val_dataset` 强制 `neg_prob = 0.0`

所以：

- 训练时会混入随机负样本；
- 验证时只看真实 teacher 样本。

### 4.4 两个容易忽略的小点

#### 小点 1：`gallery_pool` 目前其实没被用上

dataset 里初始化了：

- `self.gallery_pool = [record["gallery_path"] for record in self.records]`

但当前实现里真正抽负样本并没有用这个池，而是直接从 `records` 里随机抽 index。

#### 小点 2：dataset 返回的 `ranking_mask` / `ranking_weight` 当前没有被消费

`__getitem__` 里会返回：

- `ranking_mask`
- `ranking_weight`

但 `run_epoch()` 解包 dataloader 时，把这两个位置直接写成了 `_`，也就是当前没用。

这说明它们更像是旧设计残留，真正的 ranking mask / weight 是在 `run_epoch()` 里根据 `distances` 重新计算的。

---

## 5. 第三步：VOP 模型本体到底长什么样

VOP 本体定义在：

- [vop.py](/home/lcy/Workplace/GTA-UAV/Game4Loc/game4loc/orientation/vop.py)

### 5.1 backbone 在这里扮演什么角色

VOP 自己并不负责从 RGB 图像中提取大特征。

它依赖外部 retrieval backbone：

- `DesModel`

训练时的实际调用是：

- gallery 走 `branch="img2"`
- query 走 `branch="img1"`

提取出来的是 **feature map**，而不是最后的全局 embedding。

### 5.2 VOP 头部的结构

VOP 头非常轻：

1. `gallery_map -> 1x1 conv -> hidden_dim`
2. `query_map -> 1x1 conv -> hidden_dim`
3. 两边都做 `L2 normalize`
4. 对每个候选角度：
   - 先把 `query_map` 旋转到该角度；
   - 再和 `gallery_map` 拼成 4 个通道块：
     - `gallery`
     - `query_rot`
     - `gallery * query_rot`
     - `abs(gallery - query_rot)`
5. 把这个拼接结果送进一个小 head：
   - `1x1 conv`
   - `ReLU`
   - `1x1 conv`
6. 最后在空间维度做平均，得到该角度的一个 logit。

所以本质上，VOP 不是 Transformer decoder，也不是复杂 cross-attention。  
它就是：

> 先把 query feature map 按角度旋转，再看它和 gallery feature map 在局部上匹不匹配。

### 5.3 旋转是在什么空间做的

这里旋转的不是原图，而是 **query 的 feature map**。

实现方式是：

- `affine_grid`
- `grid_sample`

也就是说：

- backbone 特征先抽出来；
- VOP 在 feature space 里按候选角度旋转 query feature；
- 再和 gallery feature 做比较。

这也是它比“每个角度都重新跑完整 retrieval backbone”更省的一点。

### 5.4 当前默认结构大小

当前主 checkpoint 元数据里：

- `hidden_dim = 128`
- `candidate_angles = 36`

因此每个样本最终会输出：

- 一个长度为 `36` 的 logits 向量；
- 经过 softmax 后就是长度为 `36` 的后验分布。

---

## 6. 第四步：训练时到底优化什么

这一块是最核心的“毛孔级”部分，因为当前并不是单一 loss，而是两套监督模式。

---

## 7. 监督模式 A：`posterior`

这是当前较早、也更接近“分布学习”原始思路的模式。

### 7.1 总体 loss 公式

在 `posterior` 模式下，总损失是：

```text
L = L_kl
  + entropy_weight * L_entropy
  + ranking_weight * L_rank
  + ce_weight * L_ce
```

默认系数是：

- `entropy_weight = 0.1`
- `ranking_weight = 0.5`
- `ce_weight = 0.5`

也就是 4 项混合监督。

### 7.2 `L_kl`：和 soft teacher 做分布对齐

这一项是：

```text
KL( log_softmax(logits), target_probs )
```

含义很直接：

- teacher 给出一个 36 维 soft distribution；
- student 输出一个 36 维 posterior；
- 用 KL 让 student 整体逼近 teacher。

这项 loss 主要负责：

- 让模型学会“整条角度分布”；
- 而不是只学一个 winner-takes-all 的角度。

### 7.3 `L_entropy`：让预测分布的“尖锐程度”也和 teacher 接近

这一项是：

```text
L1( H(pred), H(target) )
```

其中熵是做过归一化的圆周概率熵。

它的作用不是直接管“哪个角度对”，而是管：

- 预测到底该尖还是该平；
- teacher 模糊时，student 不要乱变尖；
- teacher 很明确时，student 也不要一直很平。

简单说：

- `KL` 管“形状整体接不接近”；
- `entropy loss` 管“分布浓不浓、散不散”。

### 7.4 `L_rank`：把明显更差的角度往下压

这一项只在“有信息量”的角度对上生效。

先定义哪些 pair 是 informative：

- 当前角度有有限误差；
- 这个角度不是 best angle；
- 且它和 best angle 的误差差至少达到 `ranking_gap_m`

当前默认：

- `ranking_gap_m = 5.0`

也就是说，只有“比最优角至少差了 5 米”的角度，才会进入 ranking supervision。

然后它会根据 gap 的大小生成一个权重：

```text
weight = clamp( gap / ranking_gap_scale_m, 0.25, 2.0 )
```

默认：

- `ranking_gap_scale_m = 20.0`

最后对 best logit 和这些差角度的 logit 做一个 margin ranking 风格的约束：

```text
relu( margin - (best_logit - bad_logit) )
```

其中 margin 又会和权重有关。

直观上：

- 如果一个角度比 best angle 差很多，那它更应该被压下去；
- 如果只是稍差一点，那约束就弱一些。

### 7.5 `L_ce`：只在“比较明确”的样本上做硬分类

这一项不是对所有样本都做。

只有满足下面这些条件的样本才参与：

- `best_index >= 0`
- `distance_gap` 是有限值
- `distance_gap >= ce_gap_m`
- `target_entropy <= ce_entropy_max`

当前默认：

- `ce_gap_m = 5.0`
- `ce_entropy_max = 0.8`

这意味着：

- 如果一个样本 top1 和 top2 差得不够开；
- 或者 teacher 本身太模糊；
- 那它就不会参与硬分类 CE。

这项设计非常重要，因为它体现的是：

> 只有在 teacher 足够“可信且可分”的时候，才强迫模型去硬判一个最优角。

换句话说，`L_ce` 是一个“只对清晰样本生效的硬监督”。

---

## 8. 监督模式 B：`useful_bce`

这是你们后面开始尝试的“有效角集合监督”版本。

### 8.1 它为什么被提出来

`posterior` 模式虽然更优雅，但一个问题是：

- teacher 是 softmax 后的概率分布；
- 它会强行把所有角度都压进一个概率归一化框架里；
- 但真实情况里，很多样本其实不是“唯一好角度”，而是“有一小批角度都可接受”。

所以你们后面改成了：

> 先定义一个“有效角集合”，再让模型学这个集合。

### 8.2 有效角集合怎么定义

对每个样本：

- 找到 best distance；
- 把所有满足

```text
distance <= best_distance + useful_delta_m
```

的角度都标成正类。

当前默认：

- `useful_delta_m = 5.0`

也就是说：

- 只要某个角度的最终误差没有比最佳角差超过 `5m`；
- 它就被视为一个“仍然有用的角度”。

### 8.3 `useful_bce` 优化的是什么

当前实现里，一旦 `supervision_mode == useful_bce`：

- `L_kl = 0`
- `L_entropy = 0`
- `L_rank = 0`
- `L_ce = useful BCE`

也就是说名字虽然还叫 `ce_weight`，但此时它乘的已经不是多类交叉熵，而是：

- **多标签 BCE with logits**

### 8.4 BCE 的权重怎么做

每个样本里：

- 正类角度数 = `pos_count`
- 有效角度总数 = `valid_count`
- 负类角度数 = `neg_count`

然后为正类角度设置：

```text
pos_weight = clamp( neg_count / pos_count, 1.0, 10.0 )
```

这意味着：

- 如果正类角度很少，正类会被加大权重；
- 但不会无限增大，最大限制到 `10`。

这个设计本质上是在缓解：

- 正类角度集合通常比负类角度集合小得多；
- 如果不加权，模型很容易靠“全部压低”来取巧。

### 8.5 负样本在 `useful_bce` 模式下如何工作

如果是随机负样本行：

- `best_index = -1`
- `useful_targets` 全 0
- `valid_mask` 会被强制设为全 1

这样它在 BCE 里的语义就变成：

> 对于错误的 query-gallery 配对，所有角度都应该是负类。

这点非常合理，也非常关键。

---

## 9. 训练循环到底在做什么

### 9.1 train / val 切分

当前 `teacher_0407_full.pt` 一共 `502` 个样本。

脚本默认：

- `val_ratio = 0.1`

所以会先：

- `random.shuffle(records)`
- 然后切出
  - `50` 个验证样本
  - `452` 个训练样本

注意：

- 这个切分不是按场景分层做的；
- 而是直接在 teacher record 级别打乱后切。

### 9.2 backbone 冻结策略

当前实现默认会：

- 加载 retrieval backbone；
- `backbone.eval()`
- 把所有参数 `requires_grad=False`

如果 `partial_unfreeze=last_block`，它只会：

- 把最后一个 block 的参数改成 `requires_grad=True`

但有一个很细的实现细节：

> 即便解冻了最后一个 block，当前代码也没有把 backbone 切回 `train()`。

也就是说：

- 梯度是能回传的；
- 但 backbone 仍处在 eval 模式；
- 对 Transformer 来说通常问题不算致命，但这和真正的 end-to-end finetune 还是有区别。

### 9.3 feature 提取时何时允许梯度

run_epoch 里有一段：

```python
with torch.set_grad_enabled(bool(train and train_backbone)):
    gallery_map = backbone.extract_feature_map(...)
    query_map = backbone.extract_feature_map(...)
```

这意味着：

- 如果 backbone 冻结，feature 提取不建图；
- 只有在训练阶段且允许 train_backbone 时，feature 提取才会带梯度。

### 9.4 优化器

当前训练使用：

- `AdamW`

优化参数包括：

- VOP 头的全部参数；
- 加上任何被设置为 `requires_grad=True` 的 backbone 参数。

### 9.5 checkpoint 选择规则

脚本不是按最好 `top1 hit` 保存，而是按：

- **最低的 `val_loss`**

保存 `best_state`。

也就是说：

- 保存的是验证集 loss 最低时的 VOP 头权重；
- 不是最后一个 epoch；
- 也不是 `R@1` 或 `useful_top3` 最好时的权重。

### 9.6 保存时到底存了什么

checkpoint 里保存的是：

- `state_dict`：只包含 VOP 头；
- `in_channels`
- `hidden_dim`
- `candidate_angles_deg`
- `rotate_step`
- `model`
- `checkpoint_start`
- 一系列监督超参数

但注意：

- **没有保存 optimizer 状态**
- **没有保存 epoch 数**
- **没有保存 best_val 对应的 epoch**
- **没有把 backbone 权重一起打包进去**

因此推理时需要：

1. 先单独加载 retrieval backbone；
2. 再单独加载 VOP head checkpoint。

---

## 10. 训练日志里会打印什么

每个 epoch 会打印两组统计：

- `train_*`
- `val_*`

包括：

- `loss`
- `kl`
- `rank`
- `ce`
- `top1_angle_acc`
- `useful_top1_hit`
- `useful_top3_coverage`
- `val_rank_pairs`
- `val_ce_samples`

这里面最容易误解的是两个指标：

### 10.1 `top1_angle_acc`

它是：

- `argmax(logits)` 和 `argmax(target_probs)` 是否一致

所以它本质上还是一个硬 argmax 指标。

对于多峰 teacher 来说，这个指标不是没用，但它并不能完整反映：

- 模型有没有把“第二好 / 第三好”的可用角也抬上来。

### 10.2 `useful_top3_coverage`

这个指标更贴近后续 `prior_topk` 的真实用途。

它问的是：

> 预测的 top-3 里，是否至少包含一个“有效角”？

如果从方法角度看，我认为它比单纯的 `top1_angle_acc` 更重要。

---

## 11. 当前主线里两个 checkpoint 到底有什么区别

### 11.1 `vop_0407_full_rankce_e6.pth`

从 checkpoint 元数据看，它的关键信息是：

- `model = vit_base_patch16_rope_reg1_gap_256.sbb_in1k`
- `img_size = 384`
- `rotate_step = 10.0`
- `hidden_dim = 128`
- `ranking_weight = 0.5`
- `ranking_margin = 0.2`
- `ranking_gap_m = 5.0`
- `ranking_gap_scale_m = 20.0`
- `ce_weight = 0.5`
- `ce_gap_m = 5.0`
- `ce_entropy_max = 0.8`
- `checkpoint_start = ./pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth`
- `candidate_angles = 36`
- `unfrozen_names_count = 0`

这说明这个版本的特征是：

- backbone 冻结；
- 用 `posterior` 风格的混合监督；
- 排序约束和 CE 都开着；
- 是一种“软分布 + 排序 + 部分硬监督”的折中训练方案。

注意：

- 这个 checkpoint 元数据里没有显式保存 `supervision_mode`；
- 但从参数组合和命名 `full_rankce` 看，它对应的就是当前所说的 rank+CE 方案。

### 11.2 `vop_0408_useful5_bce_e6.pth`

它的元数据更完整：

- `ranking_weight = 0.0`
- `ce_weight = 1.0`
- `filter_entropy_max = 1.0`
- `filter_gap_m = 0.0`
- `filtered_record_count = 502`
- `partial_unfreeze = none`
- `supervision_mode = useful_bce`
- `useful_delta_m = 5.0`

这说明这个版本的特征是：

- 不再用 ranking loss；
- 主监督换成了有效角集合的 BCE；
- teacher 样本没有额外过滤；
- backbone 也仍然是全冻结。

从你们当前汇报结果看，这个版本在最终定位误差上确实更好，说明：

- 训练目标的设计，已经开始真实影响 VOP 的使用效果；
- 这不是简单“多跑几次”造成的偶然结果。

---

## 12. 当前训练过程到底学到了什么

这个问题很重要，因为它决定你汇报时该怎么定义 VOP。

当前 VOP 训练真正学到的是：

- 给定一个 query-gallery 对；
- 在离散角度集合里；
- 哪些角度更可能导向较好的最终定位结果。

它没有直接学到：

- 稀疏匹配本身怎么更稳；
- RANSAC 怎么更强；
- 单应矩阵怎么更准；
- 检索 backbone 怎么更强。

所以从研究表述上，它更像：

> 一个基于检索特征的方向打分器 / 方向后验头

而不是一个“替代整个细定位器”的模型。

---

## 13. 这套训练过程的局限和风险

如果要“毛孔级”讲清楚，局限也必须一起写。

### 13.1 teacher 是由当前 matcher 反推出来的

这意味着：

- teacher 不是绝对真值；
- 它会继承当前 matcher 的偏差；
- 如果 matcher 在某些场景本身就不稳定，teacher 也会带偏。

### 13.2 teacher 只看 phase-1 角度结果

当前 `build_vop_teacher.py` 明确只保留：

- `phase == 1`

所以 teacher 里没有显式利用 phase-2 的更细角度修正。

### 13.3 teacher 只使用第一个正样本卫星 tile

当前并没有把所有正样本 gallery 都纳入 teacher；
这会限制方向建模对多正样本情形的泛化。

### 13.4 训练增强非常弱

目前几乎只有：

- resize
- normalize

没有更强的数据增强，例如：

- 随机裁剪
- 颜色扰动
- 模糊 / 噪声

这说明当前训练更偏“稳定对齐 teacher”，而不是“强泛化训练”。

### 13.5 负样本还不够难

现在负样本只是：

- 随机换一个别的 gallery

它不是 hard negative mining。

所以当前学到的“拒绝错误配对”能力，很可能还偏弱。

### 13.6 当前保存的 best checkpoint 只看 val loss

这可能带来一个风险：

- 最低 `val_loss` 不一定对应最好的 `prior_topk` 最终定位效果。

如果后面继续做严格实验，我建议把 checkpoint 选择标准也往：

- `useful_top3_coverage`
- 或直接最终 `prior_topk` 定位表现

上靠一靠。

### 13.7 backbone 即便解冻，当前实现也不是完整 train mode

这一点前面说过，但这里再强调一次：

- `last_block` 解冻并不等于完整 end-to-end finetune；
- 当前实现更像“在 eval 态 backbone 上允许最后一层回传梯度”。

这个细节如果后面真的要做 backbone 联训，建议明确重构。

---

## 14. 如果要把这段训练过程讲给导师，最适合怎么说

我建议你可以用下面这段逻辑：

1. 我们先不直接学习一个角度标签，而是先离线枚举 36 个角度；
2. 对每个角度都真实跑一次后续定位，得到最终定位误差；
3. 用这条角度-误差曲线构造 teacher；
4. 再训练一个轻量方向头，让它只看 query-gallery 的检索特征，就预测哪几个角度最值得试；
5. 推理时不再把所有角度都穷举一遍，而是只保留 top-k 个角度进入真实匹配器。

如果导师继续追问“那这个训练到底学到了什么”，你可以直接说：

> 它学到的不是最终几何本身，而是一个和最终定位质量相关的方向先验。

这个说法是准确的，而且和代码实现一致。

---

## 15. 一段更接近当前主线的复现实验模板

下面这两条命令不是我从日志里逐字符抄出来的“唯一真实命令”，但和当前代码、teacher cache、checkpoint 元数据是一致的。

### 15.1 构造 teacher cache

```bash
/home/lcy/miniconda3/envs/gtauav/bin/python /home/lcy/Workplace/GTA-UAV/Game4Loc/build_vop_teacher.py \
  --data_root /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset \
  --pairs_meta_file same-area-drone2sate-train.json \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --img_size 384 \
  --checkpoint_start ./pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth \
  --rotate_step 10 \
  --temperature_m 25 \
  --output_path ./work_dir/vop/teacher_0407_full.pt
```

### 15.2 训练 rank+CE 风格版本

```bash
/home/lcy/miniconda3/envs/gtauav/bin/python /home/lcy/Workplace/GTA-UAV/Game4Loc/train_vop.py \
  --teacher_cache ./work_dir/vop/teacher_0407_full.pt \
  --output_path ./work_dir/vop/vop_0407_full_rankce_e6.pth \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --img_size 384 \
  --checkpoint_start ./pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth \
  --hidden_dim 128 \
  --ranking_weight 0.5 \
  --ranking_margin 0.2 \
  --ranking_gap_m 5.0 \
  --ranking_gap_scale_m 20.0 \
  --ce_weight 0.5 \
  --ce_gap_m 5.0 \
  --ce_entropy_max 0.8
```

### 15.3 训练 useful-angle BCE 版本

```bash
/home/lcy/miniconda3/envs/gtauav/bin/python /home/lcy/Workplace/GTA-UAV/Game4Loc/train_vop.py \
  --teacher_cache ./work_dir/vop/teacher_0407_full.pt \
  --output_path ./work_dir/vop/vop_0408_useful5_bce_e6.pth \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --img_size 384 \
  --checkpoint_start ./pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth \
  --hidden_dim 128 \
  --supervision_mode useful_bce \
  --useful_delta_m 5.0 \
  --ranking_weight 0.0 \
  --ce_weight 1.0 \
  --filter_entropy_max 1.0 \
  --filter_gap_m 0.0 \
  --partial_unfreeze none
```

---

## 16. 最后一句总结

如果把当前 VOP 训练过程压缩成一句最准确的话，那就是：

> 我们先用真实定位误差构造一个“角度质量分布” teacher，再训练一个轻量方向头，让它仅凭 query-gallery 检索特征去逼近这个分布，从而在推理时只保留少量高价值角度进入细定位。

这句话既贴近代码，也适合拿去给导师讲。
