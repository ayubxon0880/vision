from ultralytics import YOLO
import cv2

# YOLO modelni yaratish
model = YOLO('best250.pt')

# Video olish
video = 'test_video.mp4'
cap = cv2.VideoCapture(video)

# Ko'zni aniqlash
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Ko'zni aniqlash
    results = model(frame)

    # Aniqlangan ko'zlarni chizish
    for pred in results.pred:
        for det in pred:
            xmin, ymin, xmax, ymax, conf, cls = det
            label = model.names[int(cls)]
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Ekranga chiqarish
    cv2.imshow('Frame', frame)
    
    # 'q' tugmasi orqali chiqish
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tozalash
cap.release()
cv2.destroyAllWindows()
