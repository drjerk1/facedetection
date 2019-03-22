from errors import *
from signals import *
from glob_const import *

try:
    from tensorflow.python.keras.layers import Conv2D, Input, ZeroPadding2D, Dense
    from tensorflow.python.keras.models import Model
    from fdmobilenet import FDMobileNet
    from facemobilenet import FaceMobileNet
    from resnet18 import ResNet18SI
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
supported_models = ( "facedetection-mobilenetv2", "facedetection-fdmobilenet", "genderestimation-facemobilenet", "emotionsrecognision-resnet18" )

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

def create_tf_emotionsrecognision_resnet18(size, alpha):
    input_tensor = Input(shape=(size, size, 1))
    output_tensor = ResNet18SI(input_tensor, alpha, 0)
    output_tensor = Dense(7)(output_tensor)

    return Model(inputs=input_tensor, outputs=output_tensor)

def create_tf_genderestimation_facemobilenet(size, alpha):
    input_tensor = Input(shape=(size, size, 3))
    output_tensor = FaceMobileNet(input_tensor, alpha=alpha)
    output_tensor = Dense(2)(output_tensor)

    return Model(inputs=input_tensor, outputs=output_tensor)

def create_model_tf(name, size, alpha):
    if name == supported_models[1]:
        model = create_tf_facedetection_fdmobilenet(size, alpha)
    elif name == supported_models[0]:
        model = create_tf_facedetection_mobilenetv2(size, alpha)
    elif name == supported_models[2]:
        model = create_tf_genderestimation_facemobilenet(size, alpha)
    elif name == supported_models[3]:
        model = create_tf_emotionsrecognision_resnet18(size, alpha)
    else:
        raise ErrorSignal(unknown_model_error)
    weights_path = tf_weights_prefix + name + "-size" + str(int(size)) + "-alpha" + str(float(alpha)) + ".h5"
    try:
        model.load_weights(weights_path)
    except IOError:
        raise ErrorSignal(unknown_model_error)
    return model

class FaceDetectionOpenVINOModel():
    def __init__(self, input_name, model, num_shots, grids):
        self.input_name = input_name
        self.model = model
        self.num_shots = num_shots
        self.grids = grids

    def predict(self, batch, batch_size=None, verbose=None):
        if batch.shape[0] != self.num_shots:
            raise ErrorSignal(first_shape_mismatch)

        batch = np.transpose(batch, (0, 3, 1, 2))
        infer = self.model.infer(inputs={self.input_name: batch})
        predictions = infer[list(infer.keys())[0]]
        predictions = np.reshape(predictions, (batch.shape[0], 5, self.grids, self.grids))
        predictions = np.transpose(predictions, (0, 2, 3, 1))

        return predictions

class GenderEstimationOpenVINOModel():
    def __init__(self, input_name, model, num_shots=None, grids=None):
        self.input_name = input_name
        self.model = model

    def predict(self, batch, batch_size=None, verbose=None):
        batch = np.transpose(batch, (0, 3, 1, 2))
        predictions = []

        for i in range(batch.shape[0]):
            infer = self.model.infer(inputs={self.input_name: np.array([batch[i]])})
            prediction = infer[list(infer.keys())[0]][0].reshape((2,))
            predictions.append(prediction)

        predictions = np.array(predictions)
        return predictions

class EmotionsRecognisionOpenVINOModel():
    def __init__(self, input_name, model, num_shots=None, grids=None):
        self.input_name = input_name
        self.model = model

    def predict(self, batch, batch_size=None, verbose=None):
        batch = np.transpose(batch, (0, 3, 1, 2))
        predictions = []

        for i in range(batch.shape[0]):
            infer = self.model.infer(inputs={self.input_name: np.array([batch[i]])})
            prediction = infer[list(infer.keys())[0]][0].reshape((7,))
            predictions.append(prediction)

        predictions = np.array(predictions)
        return predictions

def create_model_openvino(name, size, alpha, precision, device, num_shots, grids):
    if name in supported_models:
        weight_path = openvino_prefix + name + "-size" + str(int(size)) + "-alpha" + str(float(alpha)) + "-fp" + str(int(precision)) + "-ns" + str(int(num_shots)) + ".bin"
        xml_path = openvino_prefix + name + "-size" + str(int(size)) + "-alpha" + str(float(alpha)) + "-fp" + str(int(precision)) + "-ns" + str(int(num_shots)) + ".xml"

        try:
            plugin = IEPlugin(device, plugin_dirs=None)
        except:
            raise ErrorSignal(openvino_error)

        try:
            network = IENetwork.from_ir(model=xml_path, weights=weight_path)
        except IOError:
            raise ErrorSignal(unknown_model_error)
        except:
            raise ErrorSignal(openvino_error)

        try:
            input_name = next(iter(network.inputs))
            model = plugin.load(network=network)
        except:
            raise ErrorSignal(openvino_error)

        del network

        if name.split('-')[0] == facedetection_prefix:
            return FaceDetectionOpenVINOModel(input_name, model, num_shots, grids)
        if name.split('-')[0] == genderestimation_prefix:
            return GenderEstimationOpenVINOModel(input_name, model, num_shots, grids)
        if name.split('-')[0] == emotionsrecognision_prefix:
            return EmotionsRecognisionOpenVINOModel(input_name, model, num_shots, grids)
    else:
        raise ErrorSignal(model_notsupported_by_openvino)

def create_model(inferencer, name, size, alpha, precision, device, num_shots, grids):
    if inferencer == 'tensorflow':
        return create_model_tf(name, size, alpha)
    elif inferencer == 'openvino':
        return create_model_openvino(name, size, alpha, precision, device, num_shots, grids)
    else:
        raise ErrorSignal(unknown_inferencer)
