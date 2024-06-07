from ultralytics import YOLO
import cv2
from datetime import datetime
import os

camera1 = 'rtsp://admin:hM571632@192.168.1.64/Streaming/Channels/102'
camera2 = 'rtsp://admin:hM571632@192.168.1.19/Streaming/Channels/102'
camera3 = 'rtsp://admin:hM571632@89.236.205.246:554/Streaming/Channels/101'
video = 'test_video/bazar2.mp4'

count = 0
id_list = []
total_list = []
folder_name = 'txt/'+str(datetime.now().strftime('%d.%m.%Y'))
if not os.path.exists(folder_name):
     os.mkdir(folder_name)
file_path = folder_name+'/'+str(datetime.now().strftime('%d.%m.%Y %H.%M.%S'))+'.txt'
file = open(file_path, 'w')
file.close

# Configure the tracking parameters and run the tracker
model = YOLO('best.pt')
results = model.track(
    video,
    line_width=1, 
    persist=True,
    classes=0,
    imgsz=1920,
    conf=0.5,
    iou=0.5,
    show=True,
    stream=True)

# Create folder for screenshots
screenshot_folder = 'screenshots/' + datetime.now().strftime('%d.%m.%Y')
os.makedirs(screenshot_folder, exist_ok=True)

# Open text file
file_path = f'{folder_name}/{datetime.now().strftime("%d.%m.%Y %H.%M.%S")}.txt'
with open(file_path, 'w') as file:
    pass  # Just to create the file

def save_screenshot(frame, object_id):
    screenshot_path = f'{screenshot_folder}/{datetime.now().strftime("%H.%M.%S")}_{object_id}.jpg'
    cv2.imwrite(screenshot_path, frame)
    print(f'Screenshot saved: {screenshot_path}')

old_id = 0

for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
        
        local_time = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
        check_folder = 'txt/' + datetime.now().strftime('%d.%m.%Y')
        if not os.path.exists(check_folder):
            os.mkdir(check_folder)
            file_path = check_folder+'/'+str(datetime.now().strftime('%d.%m.%Y %H.%M.%S'))+'.txt'
            total_list = []
        for i in boxes:
                # cls = str(i.cls)
                # print(cls)
                # cls = cls[cls.found("")]
                id = i.id
                if id is not None:
                    object_id = i.id.numpy()[0]
                    if i is not None and old_id != object_id:
                        save_screenshot(r.orig_img, object_id)
                        old_id = id
                    
                if(id):
                    id_list = id.numpy()
                    for item in id_list:
                        found = any(item in sublist for sublist in total_list)
                        if found:
                            for s_list in total_list:
                                if item in s_list:
                                    index = total_list.index(s_list)
                                    break
                            total_list[index][2] = local_time
                            t1 = datetime.strptime(total_list[index][1], '%d.%m.%Y %H:%M:%S')
                            t2 = datetime.strptime(total_list[index][2], '%d.%m.%Y %H:%M:%S')
                            t3 = t2 - t1
                            total_list[index][3] = t3
                            f = open(file_path, "w")
                            
                            for sublist in total_list:
                                f.write(str(sublist) + '\n')
                            
                            f.close
                            
                        else:
                            total_list.append([item, local_time, local_time, 0])
                    old_id = id               
                
                                                                  
        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()
        

