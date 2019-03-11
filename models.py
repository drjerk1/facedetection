from errors import *

try:
    from tensorflow.python.keras.layers import Conv2D, Input, ZeroPadding2D
    from tensorflow.python.keras.models import Model
    from fdmobilenet import FDMobileNet
    from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
except ImportError:
    pass
try:
    from openvino.inference_engine import IENetwork, IEPlugin
except ImportError:
    pass

import numpy as np

tf_weights_prefix = "tf_weights/"
openvino_prefix = "openvino_models/"
facedetection_openvino_supported_models = ( "facedetection-mobilenetv2", "facedetection-fdmobilenet" )

def create_tf_facedetection_fdmobilenet(size, alpha):
    input_tensor = Input(shape=(size, size, 3))
    output_tensor = FDMobileNet(input_tensor, alpha)
    output_tensor = ZeroPadding2D()(output_tensor)
    output_tensor = Conv2D(kernel_size=(3, 3), filters=5)(output_tensor)

    return Model(inputs=input_tensor, outputs=output_tensor)

def create_tf_facedetection_mobilenetv2(size, alpha):
    input_tensor = Input(shape=(size, size, 3))
    output_tensor = MobileNetV2(weights=None, include_top=False, input_tensor=input_tensor, alpha=alpha).output
    output_tensor = ZeroPadding2D()(output_tensor)
    output_tensor = Conv2D(kernel_size=(3, 3), filters=5)(output_tensor)

    return Model(inputs=input_tensor, outputs=output_tensor)

def create_model_tf(name, size, alpha):
    if name == facedetection_openvino_supported_models[1]:
        model = create_tf_facedetection_fdmobilenet(size, alpha)
    elif name == facedetection_openvino_supported_models[0]:
        model = create_tf_facedetection_mobilenetv2(size, alpha)
    else:
        raise ValueError(unknown_model_error)
    weights_path = tf_weights_prefix + name + "-size" + str(int(size)) + "-alpha" + str(float(alpha)) + ".h5"
    try:
        model.load_weights(weights_path)
    except IOError:
        raise ValueError(unknown_model_error)
    return model

class FaceDetectionOpenVINOModel():
    def __init__(self, input_name, model, num_shots, grids):
        self.input_name = input_name
        self.model = model
        self.num_shots = num_shots
        self.grids = grids

    def predict(self, batch, batch_size=None, verbose=None):
        if batch.shape[0] != self.num_shots:
            raise ValueError(first_shape_mismatch)

        batch = np.transpose(batch, (0, 3, 1, 2))
        infer = self.model.infer(inputs={self.input_name: batch})
        predictions = infer[list(infer.keys())[0]]
        predictions = np.reshape(predictions, (batch.shape[0], 5, self.grids, self.grids))
        predictions = np.transpose(predictions, (0, 2, 3, 1))

        return predictions

def create_model_openvino(name, size, alpha, precision, device, num_shots, grids):
    if name in facedetection_openvino_supported_models:
        weight_path = openvino_prefix + name + "-size" + str(int(size)) + "-alpha" + str(float(alpha)) + "-fp" + str(int(precision)) + "-ns" + str(int(num_shots)) + ".bin"
        xml_path = openvino_prefix + name + "-size" + str(int(size)) + "-alpha" + str(float(alpha)) + "-fp" + str(int(precision)) + "-ns" + str(int(num_shots)) + ".xml"

        try:
            plugin = IEPlugin(device, plugin_dirs=None)
        except:
            raise ValueError(openvino_error)

        try:
            network = IENetwork.from_ir(model=xml_path, weights=weight_path)
        except IOError:
            raise ValueError(unknown_model_error)
        except:
            raise ValueError(openvino_error)

        try:
            input_name = next(iter(network.inputs))
            model = plugin.load(network=network)
        except:
            raise ValueError(openvino_error)

        del network

        return FaceDetectionOpenVINOModel(input_name, model, num_shots, grids)
    else:
        raise ValueError(model_notsupported_by_openvino)

def create_model(inferencer, name, size, alpha, precision, device, num_shots, grids):
    if inferencer == 'tensorflow':
        return create_model_tf(name, size, alpha)
    elif inferencer == 'openvino':
        return create_model_openvino(name, size, alpha, precision, device, num_shots, grids)
    else:
        raise ValueError(unknown_inferencer)
