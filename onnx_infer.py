import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort  # 使用onnxruntime推理用上，pip install onnxruntime-gpu==1.12.0 -i  https://pypi.tuna.tsinghua.edu.cn/simple，默认安装CPU

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class YOLOv11:
    """YOLOv11 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, imgsz=(640, 640)):
        """Initialization.

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
        print(ort.get_device())
        # Numpy dtype: support both FP32 and FP16 onnx model
        self.ndtype = np.half if self.ort_session.get_inputs()[0].type == "tensor(float16)" else np.single

        self.model_height, self.model_width = imgsz[0], imgsz[1]  # 图像resize大小

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45):
        """The whole pipeline: pre-process -> inference -> post-process.

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
        # print('det预处理时间：{:.3f}s'.format(time.time() - t1))

        # 推理 inference
        t2 = time.time()
        preds = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: im})[0]
        # print('det推理时间：{:.2f}s'.format(time.time() - t2))
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
        # print('det后处理时间：{:.3f}s'.format(time.time() - t3))
        post_time = round(time.time() - t3, 3)

        return boxes, (pre_time, det_time, post_time)

    # 前处理，包括：resize, pad, HWC to CHW，BGR to RGB，归一化，增加维度CHW -> BCHW
    def preprocess(self, img):
        """Pre-processes the input image.

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
        """Post-process the prediction.

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
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:], axis=-1), np.argmax(x[..., 4:], axis=-1)]

        # NMS filtering
        # 经过NMS后的值, np.array([[x, y, w, h, conf, cls], ...]), shape=(-1, 4 + 1 + 1)
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


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--det_model",
        type=str,
        default=r"E:\Communication_Innovation_and_Entrepreneurship_Project\PCB\runs\detect\train\weights\best.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=r"E:\Communication_Innovation_and_Entrepreneurship_Project\PCB\PCB_DATASET_YOLO\images\val2017",
        help="Path to input image",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=r"E:\Communication_Innovation_and_Entrepreneurship_Project\PCB\runs\detect\res",
        help="结果保存文件夹",
    )
    parser.add_argument("--imgsz_det", type=tuple, default=(640, 640), help="Image input size")
    parser.add_argument(
        "--classes",
        type=list,
        default=["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"],
        help="类别",
    )

    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    print("开始运行：")
    # Build model
    det_model = YOLOv11(args.det_model, args.imgsz_det)
    color_palette = np.random.uniform(0, 255, size=(len(args.classes), 3))  # 为每个类别生成调色板

    for i, img_name in enumerate(os.listdir(args.source)):
        try:
            start_time = time.time()
            # Read image by OpenCV
            img = cv2.imread(os.path.join(args.source, img_name))

            # 检测Inference
            boxes, (pre_time, det_time, post_time) = det_model(img, conf_threshold=args.conf, iou_threshold=args.iou)
            print(
                f"{i + 1}/{len(os.listdir(args.source))} ==>总耗时间: {time.time() - start_time:.3f}s, 其中, 预处理: {pre_time:.3f}s, 推理: {det_time:.3f}s, 后处理: {post_time:.3f}s, 识别{len(boxes)}个目标"
            )

            for *box, conf, cls_ in boxes:
                cv2.rectangle(
                    img,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color_palette[int(cls_)],
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    f"{args.classes[int(cls_)]}: {conf:.3f}",
                    (int(box[0]), int(box[1] - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imwrite(os.path.join(args.out_path, img_name), img)

        except Exception as e:
            print(e)
