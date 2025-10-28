import base64
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, ImageDraw, ImageFont
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# 配置上传文件夹
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}

# 确保文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# PCB缺陷类别
PCB_CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
PCB_CLASSES_CN = ["缺失孔", "鼠咬", "开路", "短路", "毛刺", "多余铜"]


class YOLOv11:
    """YOLOv11 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, imgsz=(640, 640)):
        """
        Initialization.

        Args:
            onnx_model (str): Path to the ONNX model.
        """
        # 构建onnxruntime推理引擎
        self.ort_session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"],
        )
        print(f"ONNX Runtime device: {ort.get_device()}")

        # Numpy dtype: support both FP32 and FP16 onnx model
        self.ndtype = np.half if self.ort_session.get_inputs()[0].type == "tensor(float16)" else np.single

        self.model_height, self.model_width = imgsz[0], imgsz[1]  # 图像resize大小

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45):
        """
        The whole pipeline: pre-process -> inference -> post-process.

        Args:
            im0 (Numpy.ndarray): original input image.
            conf_threshold (float): confidence threshold for filtering predictions.
            iou_threshold (float): iou threshold for NMS.

        Returns:
            boxes (List): list of bounding boxes.
        """
        # 前处理Pre-process
        t1 = time.time()
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)
        pre_time = round(time.time() - t1, 3)

        # 推理 inference
        t2 = time.time()
        preds = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: im})[0]
        det_time = round(time.time() - t2, 3)

        # 后处理Post-process
        t3 = time.time()
        boxes = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        post_time = round(time.time() - t3, 3)

        return boxes, (pre_time, det_time, post_time)

    # 前处理，包括：resize, pad, HWC to CHW，BGR to RGB，归一化，增加维度CHW -> BCHW
    def preprocess(self, img):
        """
        Pre-processes the input image.

        Args:
            img (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """
        # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
        shape = img.shape[:2]  # original image shape
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
        left, right = round(pad_w - 0.1), round(pad_w + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 填充

        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    # 后处理，包括：阈值过滤与NMS
    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.

        Returns:
            boxes (List): list of bounding boxes.
        """
        x = preds  # outputs: predictions (1, 84, 8400)
        # Transpose the first output: (Batch_size, xywh_conf_cls, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls)
        x = np.einsum("bcn->bnc", x)  # (1, 8400, 84)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls) into one
        x = np.c_[x[..., :4], np.amax(x[..., 4:], axis=-1), np.argmax(x[..., 4:], axis=-1)]

        # NMS filtering
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        # 重新缩放边界框，为画图做准备
        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            return x[..., :6]  # boxes
        else:
            return []


# 全局模型实例
det_model = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def draw_chinese_text(img, text, position, font_size=20, color=(0, 0, 255)):
    """
    在图像上绘制中文文本.

    Args:
        img: OpenCV图像 (BGR格式)
        text: 要绘制的文本
        position: 文本位置 (x, y)
        font_size: 字体大小
        color: 文本颜色 (BGR格式)

    Returns:
        绘制文本后的图像
    """
    # 将OpenCV图像转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 尝试使用系统字体，如果失败则使用默认字体
    try:
        # Windows系统字体
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
    except:
        try:
            # 备用字体
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttf", font_size)
        except:
            # 使用默认字体
            font = ImageFont.load_default()

    # 绘制文本
    draw.text(position, text, font=font, fill=color)

    # 将PIL图像转换回OpenCV图像
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv


def init_model():
    """初始化模型."""
    global det_model
    model_path = r"E:\Communication_Innovation_and_Entrepreneurship_Project\PCB\runs\detect\train\weights\best.onnx"
    if os.path.exists(model_path):
        det_model = YOLOv11(model_path, imgsz=(640, 640))
        print("模型加载成功!")
    else:
        print(f"模型文件不存在: {model_path}")
        det_model = None


@app.route("/")
def index():
    """主页."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """处理文件上传和检测."""
    if "file" not in request.files:
        return jsonify({"error": "没有选择文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "没有选择文件"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"

        # 保存上传的文件
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # 读取图像
            img = cv2.imread(filepath)
            if img is None:
                return jsonify({"error": "无法读取图像文件"}), 400

            # 获取参数
            conf_threshold = float(request.form.get("conf_threshold", 0.25))
            iou_threshold = float(request.form.get("iou_threshold", 0.6))

            # 检测
            if det_model is None:
                return jsonify({"error": "模型未加载"}), 500

            boxes, (pre_time, det_time, post_time) = det_model(
                img, conf_threshold=conf_threshold, iou_threshold=iou_threshold
            )

            # 绘制结果
            result_img = img.copy()
            color_palette = np.random.uniform(0, 255, size=(len(PCB_CLASSES), 3))

            detection_results = []
            for *box, conf, cls_ in boxes:
                # 绘制边界框
                cv2.rectangle(
                    result_img,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color_palette[int(cls_)],
                    2,
                    cv2.LINE_AA,
                )

                # 添加中文标签
                label = f"{PCB_CLASSES_CN[int(cls_)]}: {conf:.3f}"
                result_img = draw_chinese_text(
                    result_img, label, (int(box[0]), int(box[1] - 30)), font_size=24, color=(255, 0, 0)
                )

                # 保存检测结果
                detection_results.append(
                    {
                        "class": PCB_CLASSES[int(cls_)],
                        "class_cn": PCB_CLASSES_CN[int(cls_)],
                        "confidence": float(conf),
                        "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    }
                )

            # 保存结果图像
            result_filename = f"result_{filename}"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            cv2.imwrite(result_path, result_img)

            # 将结果图像转换为base64
            _, buffer = cv2.imencode(".jpg", result_img)
            img_base64 = base64.b64encode(buffer).decode("utf-8")

            # 清理上传的临时文件
            os.remove(filepath)

            return jsonify(
                {
                    "success": True,
                    "detections": detection_results,
                    "detection_count": len(boxes),
                    "timing": {
                        "preprocess": pre_time,
                        "inference": det_time,
                        "postprocess": post_time,
                        "total": round(pre_time + det_time + post_time, 3),
                    },
                    "result_image": img_base64,
                    "result_filename": result_filename,
                }
            )

        except Exception as e:
            # 清理上传的临时文件
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": f"处理图像时出错: {e!s}"}), 500

    return jsonify({"error": "不支持的文件格式"}), 400


@app.route("/batch_upload", methods=["POST"])
def batch_upload_files():
    """处理批量文件上传和检测."""
    if "files" not in request.files:
        return jsonify({"error": "没有选择文件"}), 400

    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error": "没有选择文件"}), 400

    # 获取参数
    conf_threshold = float(request.form.get("conf_threshold", 0.25))
    iou_threshold = float(request.form.get("iou_threshold", 0.6))

    if det_model is None:
        return jsonify({"error": "模型未加载"}), 500

    results = []
    total_detections = 0
    total_time = 0

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = str(int(time.time()))
            filename = f"{timestamp}_{filename}"

            # 保存上传的文件
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                # 读取图像
                img = cv2.imread(filepath)
                if img is None:
                    continue

                # 检测
                boxes, (pre_time, det_time, post_time) = det_model(
                    img, conf_threshold=conf_threshold, iou_threshold=iou_threshold
                )

                # 绘制结果
                result_img = img.copy()
                color_palette = np.random.uniform(0, 255, size=(len(PCB_CLASSES), 3))

                detection_results = []
                for *box, conf, cls_ in boxes:
                    # 绘制边界框
                    cv2.rectangle(
                        result_img,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        color_palette[int(cls_)],
                        2,
                        cv2.LINE_AA,
                    )

                    # 添加中文标签
                    label = f"{PCB_CLASSES_CN[int(cls_)]}: {conf:.3f}"
                    result_img = draw_chinese_text(
                        result_img, label, (int(box[0]), int(box[1] - 30)), font_size=24, color=(255, 0, 0)
                    )

                    # 保存检测结果
                    detection_results.append(
                        {
                            "class": PCB_CLASSES[int(cls_)],
                            "class_cn": PCB_CLASSES_CN[int(cls_)],
                            "confidence": float(conf),
                            "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        }
                    )

                # 保存结果图像
                result_filename = f"result_{filename}"
                result_path = os.path.join(RESULT_FOLDER, result_filename)
                cv2.imwrite(result_path, result_img)

                # 将结果图像转换为base64
                _, buffer = cv2.imencode(".jpg", result_img)
                img_base64 = base64.b64encode(buffer).decode("utf-8")

                # 清理上传的临时文件
                os.remove(filepath)

                result_data = {
                    "filename": filename,
                    "result_filename": result_filename,
                    "detections": detection_results,
                    "detection_count": len(boxes),
                    "timing": {
                        "preprocess": pre_time,
                        "inference": det_time,
                        "postprocess": post_time,
                        "total": round(pre_time + det_time + post_time, 3),
                    },
                    "result_image": img_base64,
                }

                results.append(result_data)
                total_detections += len(boxes)
                total_time += result_data["timing"]["total"]

            except Exception:
                # 清理上传的临时文件
                if os.path.exists(filepath):
                    os.remove(filepath)
                continue

    return jsonify(
        {
            "success": True,
            "results": results,
            "total_files": len(results),
            "total_detections": total_detections,
            "total_time": round(total_time, 3),
            "average_time": round(total_time / len(results), 3) if results else 0,
        }
    )


@app.route("/download/<filename>")
def download_file(filename):
    """下载结果文件."""
    filepath = os.path.join(RESULT_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        return jsonify({"error": "文件不存在"}), 404


if __name__ == "__main__":
    print("正在初始化PCB缺陷检测系统...")
    init_model()
    print("启动Web服务器...")
    app.run(debug=True, host="0.0.0.0", port=5000)
