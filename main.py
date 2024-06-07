import cv2
import numpy as np

# YOLO klaslari
classes = ["ko'z"]

# YOLO modeli uchun konfiquratsiya va o'qishlar
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Layer nomlari
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Ranglarni tanlash
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Video faylini ochish
cap = cv2.VideoCapture('test_video/bazar1.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLO uchun rasmni o'lchash
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    # Modelga o'qish
    net.setInput(blob)
    outs = net.forward(output_layers)

    # YOLO natijalarini tekshirish
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Agar ko'z klasini aniqlasa
                # Ob'ektni kordinatalarini olish
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Koordinatalar orqali kenglik va balandlik aniqlash
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Ob'ektni kvadrat bilan belgilash
                cv2.rectangle(frame, (x, y), (x + w, y + h), colors[class_id], 2)
                cv2.putText(frame, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)

    # Tasvirni ekranga chiqarish
    cv2.imshow('frame', frame)

    # "q" tugmasini bosilganda chiqish
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Hammasini to'xtatish
cap.release()
cv2.destroyAllWindows()
