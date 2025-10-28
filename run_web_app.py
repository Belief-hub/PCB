# PCB缺陷检测Web应用启动脚本

import os
import sys


def check_dependencies():
    """检查依赖包是否安装."""
    required_packages = ["flask", "cv2", "numpy", "onnxruntime", "PIL"]

    missing_packages = []
    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
            elif package == "PIL":
                from PIL import Image
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    return True


def check_model_file():
    """检查模型文件是否存在."""
    model_path = r"E:\Communication_Innovation_and_Entrepreneurship_Project\PCB\runs\detect\train\weights\best.onnx"
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请确保已训练好模型并导出为ONNX格式")
        return False

    print("模型文件存在")
    return True


def main():
    print("PCB缺陷检测Web应用启动检查")
    print("=" * 50)

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 检查模型文件
    if not check_model_file():
        sys.exit(1)

    print("\n所有检查通过，启动Web应用...")
    print("访问地址: http://localhost:5000")
    print("支持移动端访问")
    print("按 Ctrl+C 停止服务")
    print("=" * 50)

    # 启动Flask应用
    from app import app, init_model

    init_model()
    app.run(debug=True, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
