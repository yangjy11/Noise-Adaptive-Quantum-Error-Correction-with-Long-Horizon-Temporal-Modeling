# AQEC-S4 独立方法包

本目录提取了现有实现中的 AQEC-S4 网络、FOMAML 元学习和分组迁移微调。
不包含物理码空间构造、环境仿真、反馈执行、真实训练数据或原有权重。
这里的 S4 是原研究代码采用的实现，不代表上游官方参考实现。

## 直接运行

需要 Python 3.10 及以上、PyTorch 2.2 及以上、NumPy。已安装依赖时，无需安装本包：

```bash
python examples/train_synthetic.py
```

也可以先安装本包：

```bash
python -m pip install -e .
aqec-s4-demo
```

默认使用 CPU。快速验证命令：

```bash
python examples/train_synthetic.py --meta-epochs 2 --transfer-epochs 2 --output-dir artifacts/smoke
python -m unittest discover -s tests -v
```

示例依次完成：合成源任务 → FOMAML 元训练 → 保存并加载模型 →
合成目标任务微调 → 独立 query 样本评估。
默认输出位于 `artifacts/synthetic_demo/`，包括两个新生成模型和训练摘要。
合成实验仅验证学习流程，不能用来声称复现论文的物理指标或数值结果。

## 公开范围

| 文件 | 内容 |
| --- | --- |
| `src/aqec_s4_methods/model.py` | 当前网络结构、测量融合、时序层、复数输出头 |
| `src/aqec_s4_methods/meta.py` | 自适应内循环、一阶 query 梯度聚合和外循环更新 |
| `src/aqec_s4_methods/transfer.py` | 冻结测量融合、分组学习率、迁移微调 |
| `src/aqec_s4_methods/losses.py` | 原代数损失，以及独立合成任务损失 |
| `src/aqec_s4_methods/data.py` | 张量和 episode 接口、合成任务、外部 JSON 读取 |
| `src/aqec_s4_methods/checkpoints.py` | 配置与模型权重保存、加载 |

训练和迁移都接受外部 `loss_fn`。默认示例使用复数 MSE；如需原训练目标，
可显式传入 `FeedbackGRAPELoss()`。该原损失的泄漏、能量项为代数代理，
不通过物理仿真计算；其设置取自首条 metadata，需要相应设置一致的批次。
