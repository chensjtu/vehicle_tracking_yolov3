from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image

# load weights and set defaults
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=608
conf_thres=0.7
nms_thres=0.5

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()

classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

videopath = 'Task3.mp4'

import cv2
from sort import *
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

vid = cv2.VideoCapture(videopath)
mot_tracker = Sort()

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (600,360))

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
ret, frame = vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print("Video size", vw,vh)
# outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"), fourcc, 25.0, (vw,vh), True)
outvideo = cv2.VideoWriter("Output_with_vehicle.avi", fourcc, 25.0, (vw, vh), True)

frames = 0
starttime = time.time()
while(True):
    ret, frame = vid.read()
    if not ret:
        break
    # frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())
        # print(np.size(tracked_objects, 0)) the number of vehicles in the video
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        # number of unique lables
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            cls = classes[int(cls_pred)]

            if cls_pred == 2.0 or cls_pred == 5.0 or cls_pred == 7.0 or cls_pred == 6.0 or cls_pred == 67.0:
                # car2 bus5 truck7 train6 cell phone67
                # cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 2)
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), (255,255,255), 2)
                # cv2.rectangle(frame, (x1, y1-15), (x1+len(cls)*10+25, y1), color, -1)# display a retangle to empysis the rate.
                # cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                cv2.putText(frame, "vehicle" + "-" + str(int(obj_id)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                # cv2.putText(frame, )
    cv2.imshow('Stream', frame)
    outvideo.write(frame)
    frames += 1
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
cv2.destroyAllWindows()
outvideo.release()
