from tensorflow.python.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, Add, DepthwiseConv2D, Flatten, Dense
from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.python.keras.models import Model
from useful import r

def ResNet18SI(input_tensor, alpha=1.0, dropout=0):
  batch_norm_momentum = 0.99
  relu_threshold = 6.0
  batch_norm_eps = 1e-3

  def res_block(input_tensor, filters, strides):
    output_tensor = input_tensor

    output_tensor = ZeroPadding2D()(output_tensor)
    output_tensor = Conv2D(kernel_size=(3, 3), filters=filters, strides=strides, use_bias=False)(output_tensor)
    output_tensor = BatchNormalization(epsilon=batch_norm_eps, momentum=batch_norm_momentum)(output_tensor)
    output_tensor = ReLU(relu_threshold)(output_tensor)

    add = output_tensor

    output_tensor = ZeroPadding2D()(output_tensor)
    output_tensor = Conv2D(kernel_size=(3, 3), filters=filters, strides=(1, 1), use_bias=False)(output_tensor)
    output_tensor = BatchNormalization(epsilon=batch_norm_eps, momentum=batch_norm_momentum)(output_tensor)
    output_tensor = ReLU(relu_threshold)(output_tensor)

    output_tensor = Add()([output_tensor, add])

    add = output_tensor

    output_tensor = ZeroPadding2D()(output_tensor)
    output_tensor = Conv2D(kernel_size=(3, 3), filters=filters, strides=(1, 1), use_bias=False)(output_tensor)
    output_tensor = BatchNormalization(epsilon=batch_norm_eps, momentum=batch_norm_momentum)(output_tensor)
    output_tensor = ReLU(relu_threshold)(output_tensor)

    output_tensor = Add()([output_tensor, add])

    add = output_tensor

    output_tensor = ZeroPadding2D()(output_tensor)
    output_tensor = Conv2D(kernel_size=(3, 3), filters=filters, strides=(1, 1), use_bias=False)(output_tensor)
    output_tensor = BatchNormalization(epsilon=batch_norm_eps, momentum=batch_norm_momentum)(output_tensor)
    output_tensor = ReLU(relu_threshold)(output_tensor)

    output_tensor = Add()([output_tensor, add])

    return output_tensor

  output_tensor = input_tensor
  output_tensor = res_block(output_tensor, r(64 * alpha), (1, 1))
  output_tensor = res_block(output_tensor, r(128 * alpha), (2, 2))
  output_tensor = res_block(output_tensor, r(256 * alpha), (2, 2))
  output_tensor = res_block(output_tensor, r(512 * alpha), (2, 2))

  output_tensor = DepthwiseConv2D(kernel_size=(output_tensor.shape.as_list()[1], output_tensor.shape.as_list()[2]), use_bias=False)(output_tensor)
  output_tensor = BatchNormalization(epsilon=batch_norm_eps, momentum=batch_norm_momentum)(output_tensor)
  output_tensor = Flatten()(output_tensor)

  output_tensor = Dense(r(1024 * alpha), use_bias=False)(output_tensor)
  output_tensor = BatchNormalization(epsilon=batch_norm_eps, momentum=batch_norm_momentum)(output_tensor)
  output_tensor = ReLU(relu_threshold)(output_tensor)

  if dropout != 0:
    output_tensor = Dropout(dropout)(output_tensor)

  return output_tensor
