from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r"E:\Communication_Innovation_and_Entrepreneurship_Project\PCB\runs\detect\train7\weights\best.pt")  # load model
    model.predict(source=r"E:\Communication_Innovation_and_Entrepreneurship_Project\PCB\PCB_DATASET_YOLO\images\val2017\01_missing_hole_07.jpg", save=True, save_conf=True, save_txt=True, name='output')

