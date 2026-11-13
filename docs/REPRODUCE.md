# 从零复现操作指南

## 1. 下载与环境

从公开仓库下载 ZIP 并解压，或执行：

```powershell
git clone https://github.com/sharychen1121/rgbd-robot-sorting.git
cd rgbd-robot-sorting
py -3.11 -m venv D:\robot_sort_env
D:\robot_sort_env\Scripts\Activate.ps1
python -m pip install -e ".[learning]"
python -m robosort doctor
```

`doctor` 应显示 Python 3.11、主要依赖版本和 `plane_asset_exists: true`。`cuda_available: false` 不影响 CPU 运行。PyBullet 第一次安装可能需要编译工具；若安装失败，先保留完整 pip 报错，核对 Python 为 64 位 3.11，避免直接更换许多依赖。

PyCharm 中文版：打开本仓库根目录 → 文件 → 设置 → Python → 解释器 → 添加解释器 → 选择现有 → 指向 `D:\robot_sort_env\Scripts\python.exe`。新开终端后确认所用解释器；切换设置不会改变已经打开的旧终端。

Linux/macOS：用 `python3.11 -m venv .venv` 和 `source .venv/bin/activate`。仅在 Windows 验证过本文给出的版本组合；CI 同时检查 Ubuntu 与 Windows 的核心测试。

可选择 `python -m pip install -r requirements-cpu-lock.txt` 使用本地验证环境的精确包版本快照。这不是所有平台通用的锁文件，GPU 用户应按自己平台安装 PyTorch，并记录实际环境。

## 2. 先验证仿真与控制

```powershell
python -m unittest discover -s tests -v
python -m robosort demo --method oracle --output results/01-oracle --gui --plots
python -m robosort demo --method hsv --output results/02-hsv --gui --plots
```

oracle 以理想位姿诊断机器人后端。HSV 使用实际图像颜色掩码与深度，不用仿真真值定位。应查看 `summary.json` 的 `correct_placements`、`total_objects` 和 `full_success_rate`，同时查看 `results.jsonl` 的每个 outcome。关闭线程的提示是正常退出；出现 `failed` 或 `stopped` 时要读 `reason`，不能只看窗口正常关闭。

每次命令换一个新的输出目录，已有结果会拒绝覆盖。无 `--gui` 可在后台快速运行；GUI 模式通过模拟实时步进观察搬运，运行完会关闭窗口。`final_scene.png` 是任务结束状态截图。

## 3. 生成自己的训练集

如需先直接体验学习模型，可跳过训练，使用仓库附带的演示权重运行：

```powershell
python -m robosort demo --catalogue examples/verification/learned-sorting/scenes.json --method yolo_mask --weights examples/checkpoints/yolo26-pilot.pt --output results/03-bundled-learning --gui --plots
```

这份小规模权重已完成一个场景的真实任务验收，仍需更充分训练后才能用于正式方法比较。下列步骤用于生成自己的数据和训练版本。

```powershell
python -m robosort dataset --output datasets/v1 --count 600 --seed 42 --random-colors --random-lighting
python -m robosort check-dataset datasets/v1
```

必须看到 `valid: true`；否则先修数据。目录包含：

```text
datasets/v1/
  manifest.json           # 版本、种子、场景及字节校验
  segment.yaml            # 分割训练配置
  detect.yaml             # 同图像检测训练配置
  segment/images/{train,val,test}/*.png
  segment/labels/{train,val,test}/*.txt
  detect/images/{train,val,test}/*.png
  detect/labels/{train,val,test}/*.txt
```

分割标签每行是“类别 ID + 多边形顶点”。负样本图片对应空标签文件，不是缺标签。新版本配置相对于 YAML 自身定位数据，可将整个目录一起复制到组员电脑。旧版内部试运行数据可能带绝对路径；重新生成 v1 即得到可移植配置。

## 4. 训练、验证和看预测

```powershell
python -m robosort train --data datasets/v1/segment.yaml --model yolo26n-seg.pt --output artifacts/26-v1 --epochs 30 --imgsz 640 --batch 2 --device cpu --allow-download
python -m robosort validate --weights artifacts/26-v1/best.pt --data datasets/v1/segment.yaml --output artifacts/26-val --split val
python -m robosort predict --fixed --weights artifacts/26-v1/best.pt --output results/03-prediction
```

首次训练下载官方初始权重，训练结束在 `artifacts/26-v1/best.pt` 留下本项目三类的模型。训练输出与 metadata 保存精确参数和权重哈希。`predict` 输出掩码叠加图、世界坐标 JSON、选中场景清单与来源/时间 metadata，不启动机械臂运动，可先检查视觉是否合理；提供多场景清单时只展示其中第一场。若几乎没有预测，需要检查模型类别、训练曲线、分辨率与阈值；不要直接放宽所有安全检查。

只在需要恢复同一个训练任务时使用 `--resume`。比较不同设置应创建新目录。测试集 `--split test` 用于方案固定后的最终评价，不反复用它选阈值。

## 5. 学习分支分拣与消融

```powershell
python -m robosort demo --method yolo_mask --weights artifacts/26-v1/best.pt --output results/04-mask --gui --plots
python -m robosort demo --method yolo_box --weights artifacts/26-v1/best.pt --output results/05-box --gui --plots
python -m robosort demo --method hybrid --weights artifacts/26-v1/best.pt --output results/06-hybrid --gui --plots
```

先使用同一模型，比较框与掩码对定位和任务结果的影响，再比较网络版本。两者使用同一深度筛选器和控制器，只有图像区域不同。框内的近表面深度筛选有时已足够可靠，因此不预设掩码一定胜出。

## 6. 冻结随机场景并比较

```powershell
python -m robosort scenes --count 30 --seed 1000 --regime standard --output datasets/eval-standard.json
python -m robosort evaluate --catalogue datasets/eval-standard.json --methods oracle hsv yolo_box yolo_mask hybrid --weights artifacts/26-v1/best.pt --output results/standard-v1 --plots
python -m robosort summarize results/standard-v1
```

每个方法自动重建相同场景；模型加载和一次预热在任务计时外，单模型被复用。`progress.jsonl` 保留运行过程，完整结果包以 `metadata.json` 最后写入为完成标志。中断后的 progress 不能冒充完整结果，重跑使用新目录。

`evaluate` 不指定 `--methods` 时只做 oracle、HSV 后端参考；要评价深度学习，使用上面的显式学习方法列表和本地训练权重。逐轮 `initialization_s` 是控制器设置耗时；模型冷加载/预热位于 `metadata.models`。RGB-D 采集包含在总耗时内，不算模型推理耗时。

颜色、光照、拥挤分别创建新的清单：`--regime color`、`--regime lighting`、`--regime stress`。不要混成一个数字掩盖条件差异。多种子训练与正式 30 场以上评价见 [实验协议](EXPERIMENT_PROTOCOL.md)。

## 常见失败

| 现象 | 排查顺序 |
|---|---|
| `plane.urdf` 找不到 | `doctor` 检查资源存在；虚拟环境和 `ROBOSORT_ASSET_DIR` 用英文路径 |
| `best.pt` 不存在 | 先训练；使用新的 `artifacts/.../best.pt`，不要复制旧检测模型当分割模型 |
| checkpoint classes 不符 | 原始 COCO 模型尚未完成三类微调，或拿错权重 |
| CUDA 不可用 | 当前 PyTorch 可能是 CPU 版，先 `--device cpu`；不自动降级掩盖设置 |
| `no_valid_targets` | 查看 localization/association/planning 拒绝阶段和坐标叠加图 |
| 关节、碰撞、运动超时 | 先同场景 oracle；保留失败场景，不删场景提高成功率 |
| 结果目录已存在 | 使用新版本目录，代码默认保护已有实验 |

三人协作可分别负责：数据与模型、RGB-D/执行诊断、实验与报告。所有人共享同一场景清单和权重哈希，再讨论结论。
