import os

import torch
import cv2

# Importing yolo
yoloModel = torch.hub.load('ultralytics/yolov5', 'custom', path='../Models/weights/best.pt', force_reload=True)

def yoloPredict(image):
    result = yoloModel(image)
    result.save()

    print(result)
    # get most recent image result
    base_img_path = 'runs/detect/'
    dirs = os.listdir(base_img_path)
    nums = [int(d[3:]) for d in dirs if d[3:].isdigit()]
    if len(nums) == 0:
        highest = ''
    else:
        highest = max(nums)
        print(highest)
    path = f'{base_img_path}exp{highest}/image0.jpg'
    # newpath = f'static/imagesToSendBack/yolo{highest}.jpg'
    
    print(os.listdir(os.path.dirname(path)))
    # print(os.listdir(os.path.dirname(newpath)))
    # os.rename(path, newpath)

    return int(len(result.xyxy[0]) > 0), path


image_size = (254, 254)

cap = cv2.VideoCapture('2-Zenmuse_X4S_2.mp4')

if not cap.isOpened():
    print("Cannot open video")
    exit()

# read frames from video
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here

    # choosing not to resize here for demonstration purposes
    # frame = cv2.resize(frame, image_size, cv2.INTER_AREA)
    class_idx, yolo_image_path = yoloPredict(frame)

    # Display the resulting frame
    yoloframe = cv2.imread(yolo_image_path)

    cv2.imshow('output', yoloframe)
    if cv2.waitKey(1) == ord('q'):
        break
