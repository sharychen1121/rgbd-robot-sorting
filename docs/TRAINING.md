# 深度学习环节与训练设计

网络输入是 RGB 图像，输出每个实例的类别、置信度、检测框和二值掩码。深度图不输入 YOLO，而是在网络之后参与三维定位；此版本属于“RGB 实例分割 + RGB-D 几何定位”，没有宣称训练了端到端抓取网络。

主模型采用官方 YOLO26n-seg，对照采用 YOLOv8n-seg。两个模型从官方预训练权重微调到三类：`0 cube`、`1 cylinder`、`2 box`。框/掩码消融使用同一分割 checkpoint；不能拿训练质量不同的检测模型证明掩码本身有效。

## 数据与训练预算

先用 600 场景建立起点，确认学习曲线后按需要扩大到 1500–3000 场景。场景随机布局与物体朝向，可随机颜色和光照，默认包含 15% 空场景。当前每类最多一个物体，避免在本轮研究中同时引入多实例放置管理。

按独立场景划分 train/val/test，不能把同一场景的相似截图分到不同集合。数据版本、种子、相机分辨率、光照与标签限制写入 manifest；场景掩码是自动标注工具，运行模型预测时禁止读取。

建议起点 30 epochs、640 输入、batch 2、seed 42，CPU 支持；这不保证已收敛。版本比较统一数据、种子、分辨率和预算，记录参数量与实际推理时间。如果 CPU 时间不足，可先 320 输入确认训练接口，再用正式设置重训，不能把两个预算混作公平对比。超参数与框/分割损失曲线均保存在 Ultralytics fit 输出中。

```powershell
python -m robosort train --data datasets/v1/segment.yaml --model yolo26n-seg.pt --output artifacts/26-v1 --epochs 30 --imgsz 640 --batch 2 --device cpu --allow-download
python -m robosort train --data datasets/v1/segment.yaml --model yolov8n-seg.pt --output artifacts/8-v1 --epochs 30 --imgsz 640 --batch 2 --device cpu --allow-download
```

`--device auto` 按实际 CUDA 可用性选择；显式指定 `--device 0` 但没有 CUDA 会报错，不会隐式改成 CPU。4GB 显存可先尝试 nano 模型、batch 2，显存不足则减 batch 并对所有比较模型统一该设置。

## 验证与测试

在验证集检查 box/mask Precision、Recall、mAP50、mAP50-95，以及空图误检和不同颜色的样例。阈值 `--conf` 在验证集选定；Hybrid 的中高置信度阈值也需要记录。最后在未参与选择的测试集运行 `validate --split test`，保存原始指标，不只展示最好看的截图。

定位还需单独诊断。深度反投影采用真实投影/视图矩阵、米制光轴深度和像素中心坐标；腐蚀预测区域后，选较近的表面深度带并取三维中位位置，盒子/方块可从平面主轴估计朝向。它是可解释几何估计，不保证可执行，因此完整实验还必须保留控制失败。

`localization-evaluate` 使用同一帧和同一组预测比较框与掩码定位；评分端按 bbox IoU 做全局一对一匹配，不用真实类别修正预测。输出 XY/Z/三维误差的均值、中位数、P95，同时报告匹配数量、分类正确、漏检、重复误检和全部目标上的有效定位覆盖率。误差以匹配且定位有效的预测为条件，覆盖率以全部目标为分母，二者必须同时解读。标准场景使用物体真实顶面中心评分；这是诊断真值，不参与预测。

局部测试也用解析投影与仿真表面检查反投影。正式研究若要声称“定位更准确”，仍需要扩大冻结场景数量与条件；不能只展示成功匹配样本。

## 工程输出与复现边界

训练写 `metadata.json`（参数、版本、数据/权重哈希）、`fit/`（训练原始文件）以及便捷副本 `best.pt`。推理和评测不自动下载权重；模型类名与任务不匹配会在运动前拒绝。

仓库保存代码、协议和小规模真实验证记录。大型生成数据、预训练权重、训练日志默认不纳入 Git；通过固定命令和种子重建。PyTorch、硬件与渲染器差异可能造成数值差异，数据哈希用于确认输入是否相同，不保证跨平台训练参数逐字节一致。

官方资料：[YOLO26](https://docs.ultralytics.com/models/yolo26/)、[实例分割](https://docs.ultralytics.com/tasks/segment/)、[训练配置](https://docs.ultralytics.com/modes/train/)。网络和预训练权重遵循 Ultralytics 许可，本项目整体采用 AGPL-3.0-or-later。
