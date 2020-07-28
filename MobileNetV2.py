import tensorflow.keras.backend as backend
from RoiPoolingConv import RoiPoolingConv
from tensorflow.keras.layers import Conv2D, Add, BatchNormalization, ReLU, ZeroPadding2D, Activation, \
    DepthwiseConv2D, TimeDistributed, AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    len_shape = len(backend.int_shape(inputs))
    if len_shape == 4:
        img_dim = 1
    else:
        img_dim = 2

    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    channel_axis = -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(x, 3), name=prefix + 'pad')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False,
                        padding='same' if stride == 1 else 'valid',
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(axis=channel_axis,
                           epsilon=1e-3,
                           momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1,
               padding='same',
               use_bias=False,
               activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(axis=channel_axis,
                           epsilon=1e-3,
                           momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x


def nn_base(img_input, alpha=1.0, pooling='avg'):
    channel_axis = -1

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = ZeroPadding2D(padding=correct_pad(img_input, 3),
                      name='Conv1_pad')(img_input)
    x = Conv2D(first_block_filters,
               kernel_size=3,
               strides=(2, 2),
               padding='valid',
               use_bias=False,
               name='Conv1')(x)
    x = BatchNormalization(axis=channel_axis,
                           epsilon=1e-3,
                           momentum=0.999,
                           name='bn_Conv1')(x)
    x = ReLU(6., name='Conv1_relu')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=9)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=12)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                            expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=15)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                            expansion=6, block_id=16)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = Conv2D(last_block_filters,
               kernel_size=1,
               use_bias=False,
               name='Conv_1')(x)
    x = BatchNormalization(axis=channel_axis,
                           epsilon=1e-3,
                           momentum=0.999,
                           name='Conv_1_bn')(x)
    x = ReLU(6., name='out_relu')(x)

    return x


def _inverted_res_block_td(inputs, expansion, stride, alpha, filters, block_id):
    channel_axis = -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = TimeDistributed(
            Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, activation=None,
                   name=prefix + 'expand'))(x)
        x = TimeDistributed(
            BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN'))(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(x, 3), name=prefix + 'pad')(x)
    x = TimeDistributed(DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False,
                                        padding='same' if stride == 1 else 'valid',
                                        name=prefix + 'depthwise'))(x)
    x = TimeDistributed(
        BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN'))(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = TimeDistributed(Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, activation=None,
                               name=prefix + 'project'))(x)
    x = TimeDistributed(BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999,
                                           name=prefix + 'project_BN'))(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x


def rpn(base_layer, num_anchors):
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', kernel_initializer='normal',
               name='rpn_conv1')(base_layer)
    x = BatchNormalization(axis=-1, name='rpn_bn')(x)
    x = Activation('relu', name='rpn_relu')(x)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                     name='rpn_out_classes')(x)
    x_reg = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                   name='rpn_ot_regress')(x)
    return [x_class, x_reg, base_layer]


def classifier_layers(x):
    x = _inverted_res_block_td(x, filters=160, alpha=1.0, stride=1, expansion=6, block_id=17)
    x = _inverted_res_block_td(x, filters=160, alpha=1.0, stride=1, expansion=6, block_id=18)
    x = _inverted_res_block_td(x, filters=320, alpha=1.0, stride=1, expansion=6, block_id=19)
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

    return x


def classifier(base_layers, input_rois, num_rois, nb_classes=2):
    """
    predict the class name for each input anchor and the regression of their bounding box
    :param base_layers:
    :param input_rois:
    :param num_rois:
    :param nb_classes:
    :return:
    """
    pooling_regions = 14

    x = [base_layers, input_rois]
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)(x)
    # out = classifier_layers(out_roi_pool)
    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                name='dense_class_{}'.format(nb_classes))(out)
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                               name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]
