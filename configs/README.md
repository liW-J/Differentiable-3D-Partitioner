# 配置文件说明

本目录存放 Differentiable 3D Partitioner Flow 的配置文件。

## 配置文件位置

配置文件应放在项目根目录下的 `configs/` 目录中。

## 使用方法

在代码中加载配置文件：

```python
import yaml
from pathlib import Path

# 加载配置文件
config_path = Path("configs/default.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 使用配置参数
learning_rate = config['optimizer']['learning_rate']
num_iterations = config['training']['num_iterations']
```

## 配置文件结构

配置文件采用 YAML 格式，包含以下主要部分：

- `partitioner`: Partitioner 初始化参数
- `optimizer`: 优化器配置
- `training`: 训练参数和调度配置
- `cutsize_loss`: Cutsize loss 配置
- `balance_loss`: Balance loss 配置
- `wirelength_loss`: Wirelength loss 配置
- `visualization`: 可视化配置
- `output`: 输出配置

## 示例配置文件

- `default.yaml`: 默认配置，包含所有参数的默认值
- 可以根据不同实验创建多个配置文件，例如：
  - `experiment1.yaml`: 实验1的配置
  - `experiment2.yaml`: 实验2的配置

