from ultralytics import YOLO
import cv2


# Load the YOLOv8 model
model = YOLO('05-04.pt')

# Export the model to TensorRT format
model.export(format="engine", device=0)

# Load the exported TensorRT model
model = YOLO('05-04.engine')

video = "test_video/bazar2.mp4" # путь к файлу с картинкой 

#model = YOLO('05-04.pt')

results = model.track(
    video,
    line_width=1, 
    persist=True,
    classes=0,
    #imgsz=1920,
    conf=0.5,
    iou=0.5,
    show=True,
    stream=True)


for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs