from tensorflow.python.keras.layers import ZeroPadding2D, DepthwiseConv2D, BatchNormalization, Conv2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from useful import r

def FDMobileNet(input_tensor, alpha=1.0):
    def SepConvBlock(input_tensor, filters, strides):
        output_tensor = input_tensor

        output_tensor = ZeroPadding2D()(output_tensor)
        output_tensor = DepthwiseConv2D(kernel_size=(3, 3), strides=strides)(output_tensor)
        output_tensor = BatchNormalization()(output_tensor)

        output_tensor = Conv2D(kernel_size=(1, 1), filters=filters)(output_tensor)
        output_tensor = BatchNormalization()(output_tensor)
        output_tensor = LeakyReLU(alpha=0.1)(output_tensor)

        return output_tensor

    output_tensor = input_tensor
    output_tensor = Conv2D(kernel_size=(3, 3), strides=(2, 2), filters=r(32 * alpha))(output_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = LeakyReLU(alpha=0.1)(output_tensor)

    output_tensor = SepConvBlock(output_tensor, r(64 * alpha), (2, 2))

    output_tensor = SepConvBlock(output_tensor, r(128 * alpha), (2, 2))
    output_tensor = SepConvBlock(output_tensor, r(128 * alpha), (1, 1))

    output_tensor = SepConvBlock(output_tensor, r(256 * alpha), (2, 2))
    output_tensor = SepConvBlock(output_tensor, r(256 * alpha), (1, 1))

    output_tensor = SepConvBlock(output_tensor, r(512 * alpha), (2, 2))
    for i in range(4):
        output_tensor = SepConvBlock(output_tensor, r(512 * alpha), (1, 1))
    output_tensor = SepConvBlock(output_tensor, r(1024 * alpha), (1, 1))

    return output_tensor
