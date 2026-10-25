# TraceGrad

[![工程检查](https://github.com/whx360/TraceGrad/actions/workflows/ci.yml/badge.svg)](https://github.com/whx360/TraceGrad/actions/workflows/ci.yml)

**基于 NumPy 的轻量级 CPU 自动微分引擎：从计算图、反向传播到可复现的训练与性能实验。**

TraceGrad 自实现张量计算图、局部向量—雅可比积（VJP）、反向拓扑遍历、广播梯度还原、梯度累加、计算图生命周期、神经网络层、SGD / Adam 及训练流程。项目围绕三个可以独立核验的问题展开：**梯度是否正确、模型能否完成真实分类任务、优化是否减少了实际计算代价。**

NumPy / BLAS 提供数组与矩阵乘法，PyTorch 仅用于独立数值对照，不作为训练后端。代码和文档在 AI 辅助下完成；经典算法及参考思想见[来源说明](docs/references.md)。

## 项目特点

- **梯度可核验**：解析解、中心差分和 PyTorch 三类对照，覆盖广播、共享子图及极端数值。
- **行为有明确约定**：叶子梯度累加、非标量反传种子、图释放与保留、前向值版本检查均有测试。
- **实验保留完整证据**：固定数据划分、三随机种子、验证集选型，保留正则化未改善结果的消融记录。
- **性能来自实测**：先分析瓶颈，再比较等价计算；保留逐次计时、运行环境和计算范围。
- **便于独立运行**：无需下载数据即可演示梯度与 XOR 学习；Ubuntu / Windows 持续集成检查基本功能与数值正确性。

## 快速开始

需要 Python 3.10 或更高版本；已记录的本地实验使用 Windows / Python 3.12。

```bash
git clone https://github.com/whx360/TraceGrad.git
cd TraceGrad
python -m venv .venv
```

激活虚拟环境：

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

安装项目并运行演示、测试：

```bash
python -m pip install -e ".[experiment]"
python run.py demo
python run.py test
```

`demo` 不下载数据：先核验一个可手算的梯度，再使用本项目的网络层和 Adam 学习 XOR。XOR 仅用于功能验证，不代表真实数据上的泛化能力。

安装 CPU 版 PyTorch 后，可执行完整的独立数值核验：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python run.py verify
```

已记录的完整测试为 **85 项通过，失败、错误、跳过均为 0**。未安装 PyTorch 时，`test` 会明确跳过独立框架对照用例；`verify` 则要求安装 PyTorch。测试包括中心差分、解析样例、广播、共享计算图、生命周期、极端值、损坏输入、优化器状态及训练与导出。CI 在 Ubuntu 和 Windows 上执行无需数据集的检查，当前状态可点击顶部徽章查看。

## 最小使用示例

```python
from tracegrad import Tensor

x = Tensor([1., 2., 3.], requires_grad=True)
loss = (x * x).sum()
loss.backward()

print(loss.item())  # 14.0
print(x.grad)       # [2., 4., 6.]
```

一次训练的主要数据流：

```text
输入数组 → Tensor 运算 → 动态计算图（DAG）
                             │
                       反向拓扑遍历 / 局部 VJP
                             │
                         参数叶子梯度
                             │
                         SGD / Adam
                             │
                    验证集选择 → 冻结模型评估
```

## 数值与生命周期约定

| 项目 | 当前实现 |
|---|---|
| 数据类型与形状 | float32 / float64；标量和非空数组；矩阵乘法限定为二维 |
| 广播 | 前向遵循 NumPy 逐元素广播；反向沿扩展轴求和，还原父节点形状 |
| 反向种子 | 标量输出默认种子为 1；非标量输出必须提供同形状种子 |
| 梯度累加 | 叶子梯度累加；每轮反传重新计算中间节点的伴随量 |
| 图生命周期 | 默认反传后释放图边；再次使用需在首次反传时保留图 |
| 前向数据一致性 | 公开数据只读；版本检查拒绝使用已失效的前向操作数 |
| 数值稳定性 | 稳定 log-sum-exp 与融合交叉熵；ReLU 在 0 处取导数 0 |
| 正则化 | 显式 L2：`lambda / 2 * sum(weight**2)`，不惩罚偏置，不暗加优化器衰减 |

当前范围为 CPU 一阶反向自动微分，不包含高阶导数、通用原地操作、GPU 内核或卷积，也不承诺成熟框架的接口兼容性。导出的 NPZ 文件供推理使用，不是保存完整训练状态的续训检查点。

## 真实分类实验

数据使用 Fashion-MNIST。固定协议为 **54,000 训练 / 6,000 验证 / 10,000 官方测试**，随机种子为 17 / 29 / 43；每轮实验训练 8 个 epoch，batch size 为 256，Adam 学习率为 0.001。每次运行的 epoch 与主模型类型均只按验证集交叉熵选择，预先确定的各组配置都保留测试结果。

| 模型 | 测试准确率：均值 ± 样本标准差 | 测试 macro-F1 | 测试交叉熵 |
|---|---:|---:|---:|
| 线性 Softmax | 83.480% ± 0.202 个百分点 | 0.833431 | 0.472524 |
| 784→64→10 ReLU MLP | 86.090% ± 0.114 个百分点 | 0.860119 | 0.397045 |
| 相同 MLP，L2=1e-4 | 85.927% ± 0.060 个百分点 | 0.858358 | 0.399736 |

在这组固定预算下，L2 没有改善结果。三个种子用于描述波动，不构成统计显著性结论，也不代表充分调参后的最优结果。详见[实验协议与分析](docs/experiments.md)和[逐次分类记录](benchmarks/reference/classification_runs.csv)。

![实际实验生成的学习曲线](assets/learning_curves.png)

## 性能优化与测量边界

性能分析发现，原始 VJP 会为 `requires_grad=False` 的输入图像计算随后丢弃的梯度。按父节点需求选择性计算 VJP 后，固定 MLP 前向与反向计算的中位耗时从 **1.68663 ms 降至 1.06237 ms，倍率为 1.5876×**；损失与所有参数梯度保持一致。

融合交叉熵在四组 float64 输入形状上的实测倍率为 **2.2602–3.5826×**。其中，256×100 情况的计算图节点数从 12 降至 2，图保留数组字节数从 1,034,272 降至 411,656。

测量固定使用一个 BLAS 线程，每组 15 对样本、每个样本计时 10 次内部调用。MLP 测量不包含优化器更新、数据加载或验证；图保留数组字节数也不是进程峰值 RSS。以上结果不能外推为完整训练提速，也不代表快于 PyTorch。可核查[原始计时样本](benchmarks/reference/performance_samples.csv)和[结构化结果](benchmarks/reference/performance.json)。

## 复现实验

```bash
# 下载并校验原始数据
python run.py fetch

# 快速冒烟运行，仅验证流程
python run.py experiment --quick

# 正式固定协议实验
python run.py experiment

# 独立运行性能测量
python run.py benchmark

# 完整核验后，生成本地实验报告
python run.py verify
python run.py report
```

数据集、训练模型、运行产物和生成报告均由 Git 忽略。程序拒绝覆盖已有非空实验目录；再次实验可指定新目录，例如 `python run.py experiment --output results/reproduction_01`。`report` 读取标准路径 `results/formal_v1`，快速运行结果不作为正式实验指标。数据来源、作者校验值与划分标识见[数据溯源记录](benchmarks/reference/data_sources.json)。

## 工程目录

```text
src/tracegrad/         自动微分、网络层、优化器、数据、训练与基准测量
tests/                数学、数值、状态与集成测试
configs/              固定实验配置
benchmarks/reference/ 精简实测记录、运行环境与数据溯源
docs/                 架构、数值约定、实验分析与参考来源
assets/               由实际实验生成的图表
.github/workflows/    不依赖数据下载的跨平台自动检查
```

进一步阅读：[系统架构](docs/architecture.md) · [数值约定与测试](docs/numerics.md) · [实验结果](docs/experiments.md) · [思想来源与实现边界](docs/references.md)。这些工程文档目前使用英文，入口说明以本页为准。

## 许可

项目代码采用 [MIT License](LICENSE)。数据集与依赖库分别遵守各自的许可条款。
