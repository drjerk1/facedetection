from errors import *
from signals import *
from models import create_model
from useful import r, softmax
import cv2
import numpy as np
import math
from glob_const import *
import time

default_inferencer = "tensorflow"
default_model_name = "emotionsrecognision-resnet18"
default_image_size = 48
default_model_alpha = 0.5
default_model_precision = 32
default_inference_device = "CPU"
eps = 1e-9
fill_color = 127
padding = 0.1
Y_coefs = np.array([0.299, 0.587, 0.114])

class EmotionsRecognision():
    def __init__(self, frame_w, frame_h, inferencer=None, model_name=None, image_size=None, model_alpha=None, model_precision=None, inference_device=None):
        self.frame_w = frame_w
        self.frame_h = frame_h
        try:
            self.inferencer = inferencer
            if not(self.inferencer is None):
                self.inferencer = str(self.inferencer)
            self.model_name = model_name
            if not(self.model_name is None):
                self.model_name = str(self.model_name)
                if self.model_name.split('-')[0] != emotionsrecognision_prefix:
                    raise ErrorSignal(not_valid_model_name)
            self.image_size = image_size
            if not(self.image_size is None):
                self.image_size = int(self.image_size)
            self.model_alpha = model_alpha
            if not(self.model_alpha is None):
                self.model_alpha = float(self.model_alpha)
            self.model_precision = model_precision
            if not(self.model_precision is None):
                self.model_precision = int(self.model_precision)
            self.inference_device = inference_device
            if not(self.inference_device is None):
                self.inference_device = str(self.inference_device)
        except (ValueError, TypeError):
            raise ErrorSignal(invalid_argument_value)

        if self.inferencer is None:
            self.inferencer = default_inferencer
        if self.model_name is None:
            self.model_name = default_model_name
        if self.image_size is None:
            self.image_size = default_image_size
        if self.model_alpha is None:
            self.model_alpha = default_model_alpha
        if self.model_precision is None:
            self.model_precision = default_model_precision
        if self.inference_device is None:
            self.inference_device = default_inference_device

        self.model = create_model(self.inferencer, self.model_name, self.image_size, self.model_alpha, self.model_precision, self.inference_device, 1, None)

    def recognise(self, frame, faces):
        resized_images = []

        for face in faces:
            resized_image = np.ones((self.image_size, self.image_size, 1), dtype=np.uint8) * fill_color

            face = np.array(face)

            face[0] *= self.frame_w
            face[2] *= self.frame_w
            face[1] *= self.frame_h
            face[3] *= self.frame_h

            face_w = face[2] - face[0]
            face_h = face[3] - face[1]

            face[0] -= face_w * padding
            face[1] -= face_h * padding
            face[2] += face_w * padding
            face[3] += face_h * padding

            face_w = face[2] - face[0]
            face_h = face[3] - face[1]

            if face_w > face_h:
                face[1] -= (face_w - face_h) / 2
                face[3] += (face_w - face_h) / 2
                if face[1] < 0 and face[3] < self.frame_h:
                    face[3] += -face[1]
                    face[1] = 0
                elif face[1] > 0 and face[3] > self.frame_h:
                    face[1] -= (face[3] - self.frame_h)
                    face[3] = self.frame_h
            else:
                face[0] -= (face_h - face_w) / 2
                face[2] += (face_h - face_w) / 2
                if face[0] < 0 and face[2] < self.frame_w:
                    face[2] += -face[0]
                    face[0] = 0
                elif face[0] > 0 and face[2] > self.frame_w:
                    face[0] -= (face[2] - self.frame_w)
                    face[2] = self.frame_w

            face[0] /= self.frame_w
            face[2] /= self.frame_w
            face[1] /= self.frame_h
            face[3] /= self.frame_h
            face = np.minimum(np.maximum(face, 0), 1)

            crop_x1 = r(face[0] * self.frame_w)
            crop_y1 = r(face[1] * self.frame_h)
            crop_x2 = r(face[2] * self.frame_w)
            crop_y2 = r(face[3] * self.frame_h)
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1

            if crop_w > crop_h:
                crop_h = r(crop_h / crop_w * self.image_size)
                crop_w = self.image_size
            else:
                crop_w = r(crop_w / crop_h * self.image_size)
                crop_h = self.image_size

            shift_x = r((self.image_size - crop_w) / 2)
            shift_y = r((self.image_size - crop_h) / 2)

            resized_image[shift_y:(shift_y + crop_h), shift_x:(shift_x + crop_w)] = np.dot(cv2.resize(frame[crop_y1:crop_y2, crop_x1:crop_x2], (crop_w, crop_h), interpolation=cv2.INTER_NEAREST), Y_coefs).reshape((crop_h, crop_w, 1))
            resized_images.append(resized_image)

        resized_images = np.array(resized_images, dtype=np.uint8)
        if len(resized_images) > 0:
            predictions = self.model.predict(resized_images, batch_size=min(batch_size, len(resized_images)), verbose=0)
            predictions = softmax(predictions)
        else:
            predictions = []

        return np.array(predictions)
