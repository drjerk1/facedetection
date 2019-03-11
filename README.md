# facedetection
Realtime CPU face detection using YOLOv2 algorithm with openvino and tensorflow support

Bunch of models pretrained on WIDER Face dataset provided

python main.py --help

python main.py --video myvideo.mp4

python main.py --video-descriptor 0 # capture from webcam

# tip: install Intel OpenVino, add "--inferencer openvino" to command line arguments and CPU inference will become approximatly 5x times faster
# tip: add "--num-shots 10" to command line arguments and model will detect smaller faces at the expense of speed
