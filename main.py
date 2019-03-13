from facedetection import FaceDetection
from videoreader import VideoReader
from errors import *
from signals import *
import argparse
import os

from visualiser import Visualiser
from jsonlogger import JSONLogger

width_hint = 1920
height_hint = 1080
error_prefix = "Error: "
default_mode = "byframe"

def main(video, json_logger=None, visualise_callback=False, mode=None, inferencer=None, model_name=None, image_size=None, model_alpha=None, shot_name=None, preset_name=None, prob_threshold=None, iou_threshold=None, union_threshold=None, model_precision=None, inference_device=None, num_shots_hint=None, video_descriptor=None):
    if not(mode is None):
        try:
            mode = str(mode)
        except (TypeError, ValueError):
            raise ErrorSignal(invalid_argument_value)
    else:
        mode = default_mode

    if not(json_logger is None):
        try:
            json_logger = str(json_logger)
        except (TypeError, ValueError):
            raise ErrorSignal(invalid_argument_value)

    exit_code = 1

    reader = VideoReader(video, width_hint, height_hint)
    detector = FaceDetection(reader.get_width(), reader.get_height(), inferencer=inferencer, model_name=model_name, image_size=image_size, model_alpha=model_alpha, shot_name=shot_name, preset_name=preset_name, prob_threshold=prob_threshold, iou_threshold=iou_threshold, union_threshold=union_threshold, model_precision=model_precision, inference_device=inference_device, num_shots_hint=num_shots_hint)
    callbacks = []

    if visualise_callback:
        callbacks.append(Visualiser(reader.get_width(), reader.get_height()))

    if not(json_logger is None):
        callbacks.append(JSONLogger(reader.get_width(), reader.get_height(), json_logger))

    if len(callbacks) == 0:
        reader.release()
        raise ErrorSignal(nothing_to_do)

    if mode == 'byframe':
        frame_number = 0
        try:
            while True:
                frame = reader.read()
                boxes = detector.detect(frame)
                frame_number += 1
                for callback in callbacks:
                    try:
                        callback.call(video=video, frame=frame, boxes=boxes, frame_number=frame_number)
                    except QuitSignal as ret:
                        exit_code = int(ret)
                        break
                else:
                    continue
                break
        except KeyboardInterrupt:
            exit_code = 0
            pass
        reader.release()
        for callback in callbacks:
            callback.destroy()
    elif mode == 'realtime':
        raise NotImplementedError()
    else:
        raise ErrorSignal(unknown_mode)

    return exit_code

if __name__ == "__main__":
    try:
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        parser = argparse.ArgumentParser(description='Facedetection')

        parser.add_argument('-m','--mode', help='Mode (byframe / realtime)', default=None, dest='mode')
        parser.add_argument('-j','--json-logger', help='Log results to json file', default=None, dest='json_logger')
        parser.add_argument('-l','--visualise', help='Visualuse', action='store_true', default=False, dest='visualise_callback')
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
                raise ErrorSignal(one_of_video_descr_required)
            else:
                try:
                    args['video'] = int(args['video_descriptor'])
                except ValueError:
                    raise ErrorSignal(invalid_argument_value)

        exit_code = main(**args)
        exit(exit_code)

    except ErrorSignal as e:
        print(error_prefix + str(e))
        exit(1)
