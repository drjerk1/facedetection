from signals import *
from useful import r
import cv2

quit_button = 'q'
window_prefix = "Visualising video: "
border_color = (0, 0, 255)
border_width = 2

class Visualiser():
    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h
    def call(self, **args):
        boxes = args['boxes']
        frame = args['frame']
        video = args['video']
        for i in range(len(boxes)):
            lx = r(boxes[i][0] * self.frame_w)
            ly = r(boxes[i][1] * self.frame_h)
            rx = r(boxes[i][2] * self.frame_w)
            ry = r(boxes[i][3] * self.frame_h)
            cv2.rectangle(frame, (lx, ly), (rx, ry), border_color, border_width)
        cv2.imshow(str(window_prefix + str(video)), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord(quit_button):
            raise QuitSignal(0)
    def destroy(self):
        cv2.destroyAllWindows()
