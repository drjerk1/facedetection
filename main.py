from facedetection import FaceDetection
from genderestimation import GenderEstimation
from emotionsrecognision import EmotionsRecognision
from faceid import FaceID
from videoreader import VideoReader
from errors import *
from signals import *
import argparse
import os
import numpy as np
from feauture_select import feauture_select
import json
from useful import cosine_dist_norm
from visualiser import Visualiser
from jsonlogger import JSONLogger

width_hint = 1920
height_hint = 1080
error_prefix = "Error: "
default_mode = "byframe"
faceid_dict = "faceid.json"
faceid_success_prefix = "Successfully created "
faceid_success_suffix = " faceid."

def main(video, json_logger=None, visualise_callback=False, mode=None, inferencer=None, model_name=None, image_size=None, model_alpha=None, shot_name=None, preset_name=None, prob_threshold=None, iou_threshold=None, union_threshold=None, model_precision=None, inference_device=None, num_shots_hint=None, video_descriptor=None, gender_estimation=False, emotions_recognision=False, run_faceid=False, create_feauture_list=None, faceid_search=False):
    if faceid_search:
        run_faceid = True
    if not(create_feauture_list is None):
        feauture_list = []
        run_faceid = True
        if preset_name is None:
            preset_name = 'accuracy'

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
    if gender_estimation:
        gender_estimatior = GenderEstimation(reader.get_width(), reader.get_height(), inferencer=inferencer)
    if run_faceid:
        faceid = FaceID(reader.get_width(), reader.get_height(), inferencer=inferencer)
    if emotions_recognision:
        emotions_recogniser = EmotionsRecognision(reader.get_width(), reader.get_height(), inferencer=inferencer)
    if faceid_search:
        try:
            faceid_dict_f = open(faceid_dict, "r")
        except IOError:
            raise ErrorSignal(faceid_dict_missing)
        faceid_json = json.load(faceid_dict_f)
        faceid_dict_f.close()
        faceids = []
        for name in faceid_json:
            faceids.append((name, faceid_json[name]['threshold'], np.array(faceid_json[name]['feautures'])))

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
                genders = None
                faceid_feautures = None
                emotions = None
                names = None
                if gender_estimation:
                    genders = gender_estimatior.estimate(frame, boxes)
                if emotions_recognision:
                    emotions = emotions_recogniser.recognise(frame, boxes)
                if run_faceid:
                    faceid_feautures = faceid.feautures(frame, boxes)
                if faceid_search:
                    names = []
                    for i in range(faceid_feautures.shape[0]):
                        face_names = []
                        for fid in faceids:
                            d = np.min(cosine_dist_norm(faceid_feautures[i].reshape(1, 128), fid[2]))
                            if d < fid[1]:
                                face_names.append(fid[0])
                        names.append(face_names)
                if not(create_feauture_list is None):
                    if faceid_feautures.shape[0] > 1:
                        raise ErrorSignal(only_one_face_required_in_create_feauture_list_mode)
                    elif faceid_feautures.shape[0] == 1:
                        feauture_list.append(faceid_feautures[0])
                frame_number += 1
                for callback in callbacks:
                    try:
                        callback.call(video=video, frame=frame, boxes=boxes, frame_number=frame_number, genders=genders, emotions=emotions, faceid_feautures=faceid_feautures, names=names)
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

    if not(create_feauture_list is None):
        try:
            create_feauture_list = str(create_feauture_list)
        except (KeyError, ValueError):
            raise ErrorSignal(invalid_argument_value)
        feautures, threshold = feauture_select(feauture_list)
        try:
            faceid_dict_f = open(faceid_dict, "r")
        except IOError:
            raise ErrorSignal(faceid_dict_missing)
        faceid_json = json.load(faceid_dict_f)
        faceid_dict_f.close()
        faceid_json[create_feauture_list] = {
            'threshold': float(threshold),
            'feautures': feautures.tolist()
        }
        try:
            faceid_dict_f = open(faceid_dict, "w")
        except IOError:
            raise ErrorSignal(faceid_dict_missing)
        json.dump(faceid_json, faceid_dict_f)
        faceid_dict_f.close()
        print(faceid_success_prefix + create_feauture_list + faceid_success_suffix)

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
        parser.add_argument('-t','--iou-threshold', help='IOU threshold', default=None, dest='iou_threshold')
        parser.add_argument('-b','--union-threshold', help='Union threshold', default=None, dest='union_threshold')
        parser.add_argument('-p','--model_precision', help='Model precision', default=None, dest='model_precision')
        parser.add_argument('-d','--inference-device', help='Inference device', default=None, dest='inference_device')
        parser.add_argument('-k','--num-shots', help='Hint for automatic shot scheme detector', default=None, dest='num_shots_hint')
        parser.add_argument('-g', '--gender-estimation', help='Perform gender estimation', action='store_true', default=False, dest='gender_estimation')
        parser.add_argument('-e', '--emotions-recognision', help='Perform emotions recognision', action='store_true', default=False, dest='emotions_recognision')
        parser.add_argument('-f', '--faceid', help='Find faceid feautures', action='store_true', default=False, dest='run_faceid')
        parser.add_argument('-q', '--create-feauture-list', help='Create feauture list of face', default=None, dest='create_feauture_list')
        parser.add_argument('-w', '--faceid-search', help='Search each face in faceid', action='store_true', default=False, dest='faceid_search')

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
