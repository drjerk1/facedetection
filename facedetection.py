from errors import *
from shots import shots
from presets import presets
from models import create_model
from us import union_suppression
from nms import non_max_suppression
from useful import r, sigmoid
import cv2
import numpy as np
import math

default_inferencer = "tensorflow"
default_model_name = "facedetection-mobilenetv2"
default_image_size = 224
default_model_alpha = 0.75
default_prob_threshold = 0.4
default_iou_threshold = 0.3
default_union_threshold = 0.3
default_model_precision = 32
default_inference_device = "CPU"
reduction_rate = 32
eps = 1e-9

class FaceDetection():
    @staticmethod
    def autodetect_shotname(frame_w, frame_h, num_shots_hint):
        best_shot_names = None
        best_loss = None
        for shot_name in shots:
            aspect_ratio = shots[shot_name]["aspect_ratio"]
            c = min(frame_h, frame_w / aspect_ratio)
            slice_h_shift = r((frame_h - c) / 2)
            slice_w_shift = r((frame_w - c * aspect_ratio) / 2)
            if slice_w_shift != 0 and slice_h_shift == 0:
                loss = slice_w_shift * frame_w * 2
            elif slice_w_shift == 0 and slice_h_shift != 0:
                loss = slice_h_shift * frame_h * 2
            else:
                if slice_w_shift != 0 and slice_h_shift != 0:
                    raise ValueError(math_is_wrong_error)
                else:
                    loss = 0
            if (best_loss is None) or (loss < best_loss):
                best_loss = loss
                best_shot_names = [shot_name]
            elif loss == best_loss:
                best_shot_names.append(shot_name)
        if num_shots_hint is None:
            num_shots_hint = 0
        best_num_shots_name = None
        best_num_shots_diff = None
        for shot_name in best_shot_names:
            num_shots = len(shots[shot_name]['shots'])
            if (best_num_shots_name is None) or abs(num_shots - num_shots_hint) < best_num_shots_diff:
                best_num_shots_name = shot_name
                best_num_shots_diff = abs(num_shots - num_shots_hint)

        return best_num_shots_name

    def __init__(self, frame_w, frame_h, inferencer=None, model_name=None, image_size=None, model_alpha=None, shot_name=None, preset_name=None, prob_threshold=None, iou_threshold=None, union_threshold=None, model_precision=None, inference_device=None, num_shots_hint=None):
        try:
            self.inferencer = inferencer
            if not(self.inferencer is None):
                self.inferencer = str(self.inferencer)
            self.model_name = model_name
            if not(self.model_name is None):
                self.model_name = str(self.model_name)
            self.image_size = image_size
            if not(self.image_size is None):
                self.image_size = int(self.image_size)
            self.model_alpha = model_alpha
            if not(self.model_alpha is None):
                self.model_alpha = float(self.model_alpha)
            self.shot_name = shot_name
            if not(self.shot_name is None):
                self.shot_name = str(self.shot_name)
            self.preset_name = preset_name
            if not(self.preset_name is None):
                self.preset_name = str(self.preset_name)
            self.prob_threshold = prob_threshold
            if not(self.prob_threshold is None):
                self.prob_threshold = float(self.prob_threshold)
            self.iou_threshold = iou_threshold
            if not(self.iou_threshold is None):
                self.iou_threshold = float(self.iou_threshold)
            self.union_threshold = union_threshold
            if not(self.union_threshold is None):
                self.union_threshold = float(self.union_threshold)
            self.model_precision = model_precision
            if not(self.model_precision is None):
                self.model_precision = int(self.model_precision)
            self.inference_device = inference_device
            if not(self.inference_device is None):
                self.inference_device = str(self.inference_device)
        except (ValueError, TypeError):
            raise ValueError(invalid_argument_value)

        if self.inferencer is None:
            self.inferencer = default_inferencer
        if self.model_name is None:
            self.model_name = default_model_name
        if self.image_size is None:
            self.image_size = default_image_size
        if self.model_alpha is None:
            self.model_alpha = default_model_alpha
        if self.shot_name is None:
            try:
                if not(num_shots_hint is None):
                    num_shots_hint = int(num_shots_hint)
            except (ValueError, TypeError):
                raise ValueError(invalid_argument_value)
            self.shot_name = self.autodetect_shotname(frame_w, frame_h, num_shots_hint)
        if (self.preset_name is None) and (self.prob_threshold is None):
            self.prob_threshold = default_prob_threshold
        if (self.preset_name is None) and (self.iou_threshold is None):
            self.iou_threshold = default_iou_threshold
        if (self.preset_name is None) and (self.union_threshold is None):
            self.union_threshold = default_union_threshold
        if self.model_precision is None:
            self.model_precision = default_model_precision
        if self.inference_device is None:
            self.inference_device = default_inference_device
        try:
            if self.prob_threshold is None:
                self.prob_threshold = presets[self.preset_name]["prob_threshold"]
            if self.union_threshold is None:
                self.union_threshold = presets[self.preset_name]["union_threshold"]
            if self.iou_threshold is None:
                self.iou_threshold = presets[self.preset_name]["iou_threshold"]
        except KeyError:
            raise ValueError(invalid_preset_name)
        try:
            self.shots = shots[self.shot_name]
        except KeyError:
            raise ValueError(invalid_shot_name)

        self.grids = self.image_size // reduction_rate

        self.model = create_model(self.inferencer, self.model_name, self.image_size, self.model_alpha, self.model_precision, self.inference_device, len(self.shots), self.grids)

    def detect(self, frame):
        original_frame_shape = frame.shape

        aspect_ratio = self.shots["aspect_ratio"]
        c = min(frame.shape[0], frame.shape[1] / aspect_ratio)
        slice_h_shift = r((frame.shape[0] - c) / 2)
        slice_w_shift = r((frame.shape[1] - c * aspect_ratio) / 2)
        if slice_w_shift != 0 and slice_h_shift == 0:
            frame = frame[:, slice_w_shift:-slice_w_shift]
        elif slice_w_shift == 0 and slice_h_shift != 0:
            frame = frame[slice_h_shift:-slice_h_shift, :]
        else:
            if slice_w_shift != 0 and slice_h_shift != 0:
                raise ValueError(math_is_wrong_error)

        frames = []
        for s in self.shots["shots"]:
            frames.append(cv2.resize(frame[r(s[1] * frame.shape[0]):r((s[1] + s[3]) * frame.shape[0]), r(s[0] * frame.shape[1]):r((s[0] + s[2]) * frame.shape[1])], (self.image_size, self.image_size)))
        frames = np.array(frames)

        predictions = self.model.predict(frames, batch_size=len(frames), verbose=0)

        boxes = []
        prob = []
        shots = self.shots['shots']
        for i in range(len(shots)):
            slice_boxes = []
            slice_prob = []
            for j in range(predictions.shape[1]):
                for k in range(predictions.shape[2]):
                    p = sigmoid(predictions[i][j][k][4])
                    if not(p is None) and p > self.prob_threshold:
                        px = sigmoid(predictions[i][j][k][0])
                        py = sigmoid(predictions[i][j][k][1])
                        pw = min(math.exp(predictions[i][j][k][2] / self.grids), self.grids)
                        ph = min(math.exp(predictions[i][j][k][3] / self.grids), self.grids)
                        if not(px is None) and not(py is None) and not(pw is None) and not(ph is None) and pw > eps and ph > eps:
                            cx = (px + j) / self.grids
                            cy = (py + k) / self.grids
                            wx = pw / self.grids
                            wy = ph / self.grids
                            if wx <= shots[i][4] and wy <= shots[i][4]:
                                lx = min(max(cx - wx / 2, 0), 1)
                                ly = min(max(cy - wy / 2, 0), 1)
                                rx = min(max(cx + wx / 2, 0), 1)
                                ry = min(max(cy + wy / 2, 0), 1)

                                lx *= shots[i][2]
                                ly *= shots[i][3]
                                rx *= shots[i][2]
                                ry *= shots[i][3]

                                lx += shots[i][0]
                                ly += shots[i][1]
                                rx += shots[i][0]
                                ry += shots[i][1]

                                slice_boxes.append([lx, ly, rx, ry])
                                slice_prob.append(p)

            slice_boxes = np.array(slice_boxes)
            slice_prob = np.array(slice_prob)

            slice_boxes = non_max_suppression(slice_boxes, slice_prob, self.iou_threshold)

            for sb in slice_boxes:
                boxes.append(sb)


        boxes = np.array(boxes)
        boxes = union_suppression(boxes, self.union_threshold)

        for i in range(len(boxes)):
            boxes[i][0] /= original_frame_shape[1] / frame.shape[1]
            boxes[i][1] /= original_frame_shape[0] / frame.shape[0]
            boxes[i][2] /= original_frame_shape[1] / frame.shape[1]
            boxes[i][3] /= original_frame_shape[0] / frame.shape[0]

            boxes[i][0] += slice_w_shift / original_frame_shape[1]
            boxes[i][1] += slice_h_shift / original_frame_shape[0]
            boxes[i][2] += slice_w_shift / original_frame_shape[1]
            boxes[i][3] += slice_h_shift / original_frame_shape[0]

        return boxes
