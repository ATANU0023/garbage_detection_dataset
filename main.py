from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Train the model on the Open Images V7 dataset
results = model.train(data=".\data.yaml", epochs=20, imgsz=640)