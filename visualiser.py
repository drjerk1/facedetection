from signals import *
from errors import *
from glob_const import *
from useful import r
import cv2
import numpy as np

quit_button = 'q'
window_prefix = "Visualising video: "
default_border_color = (0, 0, 0)
border_width = 2
gender_text_offset_y = -5
gender_text_offset_x = 0
gender_text_color = (0, 0, 0)
gender_male_text = "M"
gender_female_text = "W"
emotion_text_offset_y = -5
emotion_text_offset_x = 100
emotion_text_color = (0, 0, 0)

class Visualiser():
    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h
    def call(self, **args):
        boxes = args['boxes']
        frame = args['frame']
        video = args['video']
        genders = args['genders']
        emotions = args['emotions']
        for i in range(len(boxes)):
            lx = r(boxes[i][0] * self.frame_w)
            ly = r(boxes[i][1] * self.frame_h)
            rx = r(boxes[i][2] * self.frame_w)
            ry = r(boxes[i][3] * self.frame_h)
            border_color = default_border_color
            cv2.rectangle(frame, (lx, ly), (rx, ry), border_color, border_width)
            if not(genders is None):
                gender = np.argmax(genders[i])
                if gender == male_id:
                    gender = gender_male_text
                elif gender == female_id:
                    gender = gender_female_text
                else:
                    raise ErrorSignal(math_is_wrong_error)
                cv2.putText(frame, gender, (lx + gender_text_offset_x, ly + gender_text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, gender_text_color, 2, cv2.LINE_AA)
            if not(emotions is None):
                emotion = emotions[i].argmax()
                emotion_name = emotion_names[emotion]
                cv2.putText(frame, emotion_name, (lx + emotion_text_offset_x, ly + emotion_text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, emotion_text_color, 2, cv2.LINE_AA)
        cv2.imshow(str(window_prefix + str(video)), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord(quit_button):
            raise QuitSignal(0)
    def destroy(self):
        cv2.destroyAllWindows()
