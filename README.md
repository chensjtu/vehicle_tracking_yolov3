## Introduction
## vehicle recognition and tracking based on yolo v3 and sort algorithem.
This is the final project of the machine vision course. Given that multi-object tracking (MOT) is currently the research focus, this project will open source to help related research. The basis of this project is: **YOLOv3**, multi-target recognition algorithm, Kalman filter tracking, Hungarian algorithm. (IPIL 2016) This project has basically completed the identification and tracking of vehicle targets in the video, but the robustness is not perfect.The yolov3 implementation is from [darknet](https://github.com/pjreddie/darknet). BTW, this project can work well in online network.

##To run this project you need:
- python 3.6
- numpy
- scipy
- opencv-python
- sklearn
- pytorch 0.4 or 1.x
- time
- filterpy
- matplotlib
all of these is load on the request.txt. 

##Quick start:
1.Clone this file
```bash
$ git clone https://github.com/Github-chenyang/vehicle_tracking_yolov3.git

2.You are supposed to install some dependencies before getting out hands with these codes.
```bash
$ cd vehicle_tracking_yolov3
$ pip install -r ./docs/request.txt
for user in china, you can specify pypi source to accelerate install like:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 
```

3.Download YOLOv3 parameters
```bash
$ cd config/
$ wget https://pjreddie.com/media/files/yolov3.weights
$ cd ..

4.Run demo
just run the object_tracker.py and you can realize a simple offline MOA.

here is the effect of the algorithm
![image](https://github.com/Github-chenyang/vehicle_tracking_yolov3/raw/master/docs/1.png)

if this project can help you solve the project or inspire you, could you please give me a star?
---------------------------------------------
求个star～









