from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="datasets/yolo/data.yaml",
    epochs=20,
    imgsz=640
)
