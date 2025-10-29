## PCB 缺陷检测系统（基于 YOLO11）

本项目面向 PCB 质量检测场景，基于 Ultralytics YOLO11 构建端到端的缺陷识别方案，涵盖模型训练、ONNX 部署与可视化 Web 应用。

### 核心能力
- 检测 6 类 PCB 缺陷：`missing_hole`、`mouse_bite`、`open_circuit`、`short`、`spur`、`spurious_copper`
- 提供 `PCB_DATASET_YOLO/` 数据示例（YOLO 标注格式，含 train/val）
- 一键启动的 Flask Web 界面，支持单图与批量检测、中文结果展示与下载
- 支持 GPU/CPU 推理（ONNX Runtime），轻量易部署

### 目录结构
- `PCB_DATASET_YOLO/`：示例数据集（images/labels，train2017/val2017）
- `runs/detect/train/`：训练输出目录（建议导出 `weights/best.onnx` 用于 Web 推理）
- `app.py`：ONNX 推理与 Flask 后端
- `run_web_app.py`：Web 应用启动脚本
- `templates/index.html`：Web 前端页面

---

## 快速开始

### 1) 环境安装
建议使用 Python 3.8+。

```bash
pip install ultralytics flask opencv-python onnxruntime pillow numpy
```

如需 GPU 推理，安装 `onnxruntime-gpu` 并确保 CUDA/CUDNN 可用。

### 2) 训练模型（可选）
数据已按 YOLO 标注组织在 `PCB_DATASET_YOLO/`。若需自定义训练，请先准备一个数据集配置文件（示例）：

```yaml
# pcb.yaml
path: ./PCB_DATASET_YOLO
train: images/train2017
val: images/val2017
names: [missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper]
```

使用 Ultralytics 进行训练：

```bash
yolo detect train model=yolo11s.pt data=pcb.yaml epochs=100 imgsz=640
```

训练完成后，导出 ONNX：

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=12 dynamic=True
```

将导出的 `best.onnx` 放在：
`runs/detect/train/weights/best.onnx`

> 注意：`app.py` 默认从上述路径加载 ONNX（可按需修改 `init_model()` 中的路径）。

### 3) 启动 Web 应用

```bash
python run_web_app.py
```

浏览器访问 `http://localhost:5000`：
- 选择单张或多张 PCB 图片
- 可调 `置信度阈值 / IoU 阈值`
- 查看中文可视化结果、检测统计，并下载标注图

---

## 示例结果

仓库 `results/` 目录包含若干检测示例输出，可用于比对效果：
- `results/result_*.jpg`

---

## 重要说明与致谢
- 本项目基于开源的 Ultralytics YOLO 框架进行二次开发，感谢其卓越的模型与工具链。
- 若用于商业用途，请遵循本仓库 `LICENSE` 文件与 Ultralytics 相关许可政策。

---

## 许可
本仓库遵循 `LICENSE` 中的条款。请在遵守开源协议与第三方依赖许可的前提下使用本项目。

## 引用
如本项目对您的研究或产品有帮助，建议引用 Ultralytics YOLO 及相关工作。


