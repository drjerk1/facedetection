import json

prefix = "[\n"
separator = ",\n"
suffix = "{}\n]"

class JSONLogger():
    def __init__(self, frame_w, frame_h, log_file):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.log_file = open(log_file, "w")
        self.log_file.write(prefix)
    def call(self, **args):
        boxes = args['boxes']
        frame_number = args['frame_number']
        self.log_file.write(json.dumps({'frame_number': frame_number, 'boxes': boxes}, sort_keys=True, indent=4))
        self.log_file.write(separator)

    def destroy(self):
        self.log_file.write(suffix)
        self.log_file.close()
