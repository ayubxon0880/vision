
import os
import uuid

img = "D:\\python\\test_dataset\\val\\images"
label = "D:\\python\\test_dataset\\val\\label"

img_list = sorted(os.listdir(img))
label_list = sorted(os.listdir(label))

lab = []

for i in label_list:
    lab.append(os.path.splitext(i)[0])
dup = []
for i in img_list:
    if os.path.splitext(i)[0] not in lab:
        dup.append(i)
for i in dup:
    print(i)