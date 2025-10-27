from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolo11s.yaml")  # build a new model from scratch
    model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)


    # Use the model
    model.train(data="PCB.yaml", imgsz=640, batch=16, workers=8, cache=True, epochs=100)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("ultralytics\\assets\\bus.jpg")  # predict on an image
    path = model.export(format="onnx", opset=13)  # export the model to ONNX format
