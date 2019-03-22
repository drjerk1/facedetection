from tensorflow.python.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Input, Add
from useful import r

relu_threshold = 6.0
batch_norm_eps = 1e-3
batch_norm_momentum = 0.999

def inverted_res_block(input_tensor, expansion, stride, alpha, filters):
    in_channels = input_tensor.shape.as_list()[-1]
    filters = r(filters * alpha)
    output_tensor = input_tensor

    output_tensor = Conv2D(expansion * in_channels, kernel_size=(1, 1), use_bias=False)(output_tensor)
    output_tensor = BatchNormalization(epsilon=batch_norm_eps, momentum=batch_norm_momentum)(output_tensor)
    output_tensor = ReLU(relu_threshold)(output_tensor)

    output_tensor = ZeroPadding2D()(output_tensor)
    output_tensor = DepthwiseConv2D(kernel_size=(3, 3), strides=stride, use_bias=False)(output_tensor)
    output_tensor = BatchNormalization(epsilon=batch_norm_eps, momentum=batch_norm_momentum)(output_tensor)
    output_tensor = ReLU(relu_threshold)(output_tensor)

    output_tensor = Conv2D(filters, kernel_size=(1, 1), use_bias=False)(output_tensor)
    output_tensor = BatchNormalization(epsilon=batch_norm_eps, momentum=batch_norm_momentum)(output_tensor)

    if in_channels == filters and stride == 1:
        output_tensor = Add()([input_tensor, output_tensor])
    return output_tensor

def FaceMobileNet(input_tensor, alpha=1.0):
    output_tensor = input_tensor

    output_tensor = ZeroPadding2D()(output_tensor)
    output_tensor = Conv2D(filters=r(64 * alpha), kernel_size=(3, 3), strides=(2, 2), use_bias=False)(output_tensor)
    output_tensor = BatchNormalization(epsilon=batch_norm_eps, momentum=batch_norm_momentum)(output_tensor)
    output_tensor = ReLU(relu_threshold)(output_tensor)

    output_tensor = ZeroPadding2D()(output_tensor)
    output_tensor = DepthwiseConv2D(kernel_size=(3, 3), use_bias=False)(output_tensor)
    output_tensor = BatchNormalization(epsilon=batch_norm_eps, momentum=batch_norm_momentum)(output_tensor)
    output_tensor = ReLU(relu_threshold)(output_tensor)

    output_tensor = inverted_res_block(output_tensor, filters=64, alpha=alpha, stride=2, expansion=2)
    output_tensor = inverted_res_block(output_tensor, filters=64, alpha=alpha, stride=1, expansion=2)
    output_tensor = inverted_res_block(output_tensor, filters=64, alpha=alpha, stride=1, expansion=2)
    output_tensor = inverted_res_block(output_tensor, filters=64, alpha=alpha, stride=1, expansion=2)
    output_tensor = inverted_res_block(output_tensor, filters=64, alpha=alpha, stride=1, expansion=2)

    output_tensor = inverted_res_block(output_tensor, filters=128, alpha=alpha, stride=2, expansion=4)

    output_tensor = inverted_res_block(output_tensor, filters=128, alpha=alpha, stride=1, expansion=2)
    output_tensor = inverted_res_block(output_tensor, filters=128, alpha=alpha, stride=1, expansion=2)
    output_tensor = inverted_res_block(output_tensor, filters=128, alpha=alpha, stride=1, expansion=2)
    output_tensor = inverted_res_block(output_tensor, filters=128, alpha=alpha, stride=1, expansion=2)
    output_tensor = inverted_res_block(output_tensor, filters=128, alpha=alpha, stride=1, expansion=2)
    output_tensor = inverted_res_block(output_tensor, filters=128, alpha=alpha, stride=1, expansion=2)

    output_tensor = inverted_res_block(output_tensor, filters=128, alpha=alpha, stride=2, expansion=4)

    output_tensor = inverted_res_block(output_tensor, filters=128, alpha=alpha, stride=1, expansion=2)
    output_tensor = inverted_res_block(output_tensor, filters=128, alpha=alpha, stride=1, expansion=2)

    output_tensor = Conv2D(filters=r(512 * alpha), kernel_size=(1, 1), use_bias=False)(output_tensor)
    output_tensor = BatchNormalization(epsilon=batch_norm_eps, momentum=batch_norm_momentum)(output_tensor)
    output_tensor = ReLU(relu_threshold)(output_tensor)

    output_tensor = DepthwiseConv2D(kernel_size=(output_tensor.shape.as_list()[1], output_tensor.shape.as_list()[2]), use_bias=False)(output_tensor)
    output_tensor = BatchNormalization(epsilon=batch_norm_eps, momentum=batch_norm_momentum)(output_tensor)

    output_tensor = Conv2D(filters=r(128 * alpha), kernel_size=(1, 1), use_bias=False)(output_tensor)
    output_tensor = BatchNormalization(epsilon=batch_norm_eps, momentum=batch_norm_momentum)(output_tensor)

    return output_tensor
