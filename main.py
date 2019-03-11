from facedetection import FaceDetection
from videoreader import VideoReader
from errors import *
from useful import r
import cv2
import argparse
import os

width_hint = 1920
height_hint = 1080
border_color = (0, 0, 255)
border_width = 2
quit_button = 'q'
error_prefix = "Error: "

window_prefix = "Visualising video: "

one_of_video_descr_required = "Need video input"

def main(video, inferencer=None, model_name=None, image_size=None, model_alpha=None, shot_name=None, preset_name=None, prob_threshold=None, iou_threshold=None, union_threshold=None, model_precision=None, inference_device=None, num_shots_hint=None, video_descriptor=None):
    reader = VideoReader(video, width_hint, height_hint)
    detector = FaceDetection(reader.get_width(), reader.get_height(), inferencer=inferencer, model_name=model_name, image_size=image_size, model_alpha=model_alpha, shot_name=shot_name, preset_name=preset_name, prob_threshold=prob_threshold, iou_threshold=iou_threshold, union_threshold=union_threshold, model_precision=model_precision, inference_device=inference_device, num_shots_hint=num_shots_hint)
    while True:
        frame = reader.read()
        boxes = detector.detect(frame)
        for i in range(len(boxes)):
            lx = r(boxes[i][0] * reader.get_width())
            ly = r(boxes[i][1] * reader.get_height())
            rx = r(boxes[i][2] * reader.get_width())
            ry = r(boxes[i][3] * reader.get_height())
            cv2.rectangle(frame, (lx, ly), (rx, ry), border_color, border_width)
        cv2.imshow(str(window_prefix + str(video)), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord(quit_button):
            break
    reader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        parser = argparse.ArgumentParser(description='Facedetection')

        parser.add_argument('-v','--video', help='Input video filepath', default=None, dest='video')
        parser.add_argument('-r','--video-descriptor', help='Input video opencv descriptor', default=None, dest='video_descriptor')
        parser.add_argument('-i','--inferencer', help='Inferencer', default=None, dest='inferencer')
        parser.add_argument('-n','--model-name', help='Model name', default=None, dest='model_name')
        parser.add_argument('-s','--image-size', help='Image size', default=None, dest='image_size')
        parser.add_argument('-a','--alpha', help='Model alpha', default=None, dest='model_alpha')
        parser.add_argument('-z','--shot-name', help='Shot name', default=None, dest='shot_name')
        parser.add_argument('-x','--preset-name', help='Preset name', default=None, dest='preset_name')
        parser.add_argument('-c','--prob-threshold', help='Prob threshold', default=None, dest='prob_threshold')
        parser.add_argument('-f','--iou_threshold', help='IOU threshold', default=None, dest='iou_threshold')
        parser.add_argument('-b','--union-threshold', help='Union threshold', default=None, dest='union_threshold')
        parser.add_argument('-p','--model_precision', help='Model precision', default=None, dest='model_precision')
        parser.add_argument('-d','--inference-device', help='Inference device', default=None, dest='inference_device')
        parser.add_argument('-k','--num-shots', help='Hint for automatic shot scheme detector', default=None, dest='num_shots_hint')

        args = vars(parser.parse_args())
        if args['video'] is None:
            if args['video_descriptor'] is None:
                print(error_prefix + one_of_video_descr_required)
                exit(1)
            else:
                try:
                    args['video'] = int(args['video_descriptor'])
                except ValueError:
                    raise ValueError(invalid_argument_value)
        main(**args)

    except ValueError as e:
        print(error_prefix + str(e))
        exit(1)

    exit(0)
