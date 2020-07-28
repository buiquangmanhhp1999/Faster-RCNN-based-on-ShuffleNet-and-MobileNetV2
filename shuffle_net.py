import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import numpy as np
from RoiPoolingConv import RoiPoolingConv
import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)


def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length // 32

    return get_output_length(width), get_output_length(height)


def _block(x, channel_map, bottleneck_ratio, repeat=1, groups=1, stage=1):
    """
    creates a bottleneck block containing `repeat + 1` shuffle units
    Parameters
    ----------
    x:
        Input tensor of with `channels_last` data format
    channel_map: list
        list containing the number of output channels for a stage
    repeat: int(1)
    .
        number of repetitions for a shuffle unit with stride 1
    groups: int(1)
        number of groups per channel
    bottleneck_ratio: float
        bottleneck ratio implies the ratio of bottleneck channels to output channels.
        For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
        the width of the bottleneck feature map.
    stage: int(1)
        stage number
    Returns
    -------
    """
    x = _shuffle_unit(x, input_channels=channel_map[stage - 2],
                      out_channels=channel_map[stage - 1], strides=2,
                      groups=groups, bottleneck_ratio=bottleneck_ratio,
                      stage=stage, block=1)

    for i in range(1, repeat + 1):
        x = _shuffle_unit(x, input_channels=channel_map[stage - 1],
                          out_channels=channel_map[stage - 1], strides=1,
                          groups=groups, bottleneck_ratio=bottleneck_ratio,
                          stage=stage, block=(i + 1))

    return x


def _shuffle_unit(inputs, input_channels, out_channels, groups, bottleneck_ratio, strides=2, stage=1, block=1):
    """
    creates a shuffle-unit
    Parameters
    ----------
    inputs:
        Input tensor of with `channels_last` data format
    input_channels:
        number of input channels
    out_channels:
        number of output channels
    strides:
        An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the width and height.
    groups: int(1)
        number of groups per channel
    bottleneck_ratio: float
        bottleneck ratio implies the ratio of bottleneck channels to output channels.
        For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
        the width of the bottleneck feature map.
    stage: int(1)
        stage number
    block: int(1)
        block number
    Returns
    -------
    """

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    prefix = 'stage%d/block%d' % (stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    groups = (1 if stage == 2 and block == 1 else groups)

    # 1x1 GConv
    x = _group_conv(inputs, input_channels, out_channels=bottleneck_channels, groups=groups,
                    name='%s/1x1_gconv_1' % prefix)
    x = layers.BatchNormalization(axis=bn_axis, name='%s/bn_gconv_1' % prefix)(x)
    x = layers.Activation('relu', name='%s/relu_gconv_1' % prefix)(x)

    # # channel shuffle
    x = layers.Lambda(channel_shuffle, arguments={'groups': groups}, name='%s/channel_shuffle' % prefix)(x)
    # x = channel_shuffle_ver2(groups=groups)(x)
    # DepthWise Convolution
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', use_bias=False, strides=strides,
                               name='%s/1x1_dwconv_1' % prefix)(x)
    x = layers.BatchNormalization(axis=bn_axis, name='%s/bn_dwconv_1' % prefix)(x)

    x = _group_conv(x, bottleneck_channels,
                    out_channels=out_channels if strides == 1 else out_channels - input_channels, groups=groups,
                    name='%s/1x1_gconv_2' % prefix)
    x = layers.BatchNormalization(axis=bn_axis, name='%s/bn_gconv_2' % prefix)(x)

    if strides < 2:
        ret = layers.Add(name='%s/add' % prefix)([x, inputs])
    else:
        avg = layers.AveragePooling2D(pool_size=3, strides=2, padding='same', name='%s/avg_pool' % prefix)(inputs)
        ret = layers.Concatenate(bn_axis, name='%s/concat' % prefix)([x, avg])

    ret = layers.Activation('relu', name='%s/relu_out' % prefix)(ret)

    return ret


def _group_conv(x, input_channels, out_channels, groups, kernel=1, stride=1, name=''):
    """
    grouped convolution
    Parameters
    ----------
    x:
        Input tensor of with `channels_last` data format
    input_channels:
        number of input channels
    out_channels:
        number of output channels
    groups:
        number of groups per channel
    kernel: int(1)
        An integer or tuple/list of 2 integers, specifying the
        width and height of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
    stride: int(1)
        An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the width and height.
        Can be a single integer to specify the same value for all spatial dimensions.
    name: str
        A string to specifies the layer name
    Returns
    -------
    """

    if groups == 1:
        return layers.Conv2D(filters=out_channels, kernel_size=kernel, padding='same', use_bias=False, strides=stride,
                             name=name)(x)

    # number of input channels per group
    offset = input_channels // groups
    group_list = []

    for i in range(groups):
        group = layers.Lambda(lambda inputs: inputs[:, :, :, i * offset: offset * (i + 1)],
                              name='%s/g%d_slice' % (name, i))(x)
        group_list.append(
            layers.Conv2D(int(0.5 + out_channels / groups), kernel_size=kernel, strides=stride, use_bias=False,
                          padding='same', name='%s_/g%d' % (name, i))(group))

    return layers.Concatenate(name='%s/concat' % name)(group_list)


class _group_conv2(layers.Layer):
    def __init__(self, input_channels, out_channels, groups, kernel, stride, name):
        super(_group_conv2, self).__init__()
        self.groups = groups

        offset = input_channels // groups
        self.convs = [layers.Conv2D(int(0.5 + out_channels / self.groups), kernel_size=kernel, strides=stride,
                                    use_bias=False, padding='same', name='%s_/g%d' % (name, i)) for i in
                      range(self.groups)]
        self.lambdas = [layers.Lambda(lambda x: x[:, :, :, i * offset: offset * (i + 1)],
                                      name='%s/g%d_slice' % (name, i)) for i in range(self.groups)]
        self.concat = layers.Concatenate(name='%s/concat' % name)

    def call(self, inputs, **kwargs):
        group_list = []
        for i in range(self.groups):
            group = self.lambdas[i](inputs)
            conv = self.convs[i](group)
            group_list.append(conv)

        return self.concat(group_list)


def channel_shuffle(x, groups):
    """
    Parameters
    ----------
    x:
        Input tensor of with `channels_last` data format
    groups: int
        number of groups per channel
    Returns
    -------
        channel shuffled output tensor
    """
    height, width, in_channels = x.shape.as_list()[1:]
    channels_per_group = in_channels // groups

    x = K.reshape(x, [-1, height, width, groups, channels_per_group])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
    x = K.reshape(x, [-1, height, width, in_channels])

    return x


class channel_shuffle_ver2(layers.Layer):
    def __init__(self, groups):
        super(channel_shuffle_ver2, self).__init__()
        self.groups = groups

    def build(self, input_shape):
        height, width, channels = input_shape[1:]
        print(height, width, channels)
        channels_per_group = channels / self.groups
        channels_per_group = tf.cast(channels_per_group, 'int32')
        self.reshape_ver1 = layers.Reshape([height, width, self.groups, channels_per_group])
        self.permute_dimension = layers.Permute(dims=(1, 2, 4, 3))
        self.reshape_ver2 = layers.Reshape([height, width, channels])

    def call(self, inputs, **kwargs):
        x = self.reshape_ver1(inputs)
        x = self.permute_dimension(x)
        x = self.reshape_ver2(x)
        return x


def shuffle_net(input_tensor, scale_factor=1, pooling='avg', input_shape=(224, 224, 3),
                groups=3, botteleneck_ratio=0.25):
    out_dim_stage_two = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}
    num_shuffle_units = [3, 7, 3]

    if pooling not in ['max', 'avg']:
        raise ValueError('Invalid value for pooling')

    # calculate output channels for each stage
    exp = np.insert(np.arange(0, len(num_shuffle_units)), 0, 0)
    out_channels_in_stage = 2 ** exp
    out_channels_in_stage *= out_dim_stage_two[groups]
    out_channels_in_stage[0] = 24  # fist stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if K.is_keras_tensor(input_tensor):
            img_input = input_tensor
        else:

            img_input = layers.Input(tensor=input_tensor, shape=input_shape)

    x = layers.Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False,
                      strides=(2, 2), activation='relu', name='conv1')(img_input)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='max_pool1')(x)

    # create stages containing shuffle-net units beginning at stage 2
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = _block(x, out_channels_in_stage, repeat=repeat, bottleneck_ratio=botteleneck_ratio,
                   groups=groups, stage=stage + 2)

    return x


def rpn(base_layer, num_anchors, input_channels=960, bottleneck_ratio=0.25, output_channels=512, groups=3,
        kernel_size=3):
    # x = layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', kernel_initializer='normal',
    #                   name='rpn_conv1')(base_layer)
    # 1x1 GConv
    bottleneck_channels = int(output_channels * bottleneck_ratio)
    x = _group_conv(base_layer, input_channels, out_channels=bottleneck_channels, groups=groups, kernel=kernel_size,
                    name='rpn_conv1')

    x = layers.BatchNormalization(axis=-1, name='rpn_bn')(x)
    x = layers.Activation('relu', name='rpn_relu')(x)

    # channel shuffle
    # x = layers.Lambda(channel_shuffle, arguments={'groups': groups}, name='rpn_channel_shuffle')(x)

    x_class = layers.Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                            name='rpn_out_classes')(x)
    x_reg = layers.Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero',
                          name='rpn_ot_regress')(x)

    return [x_class, x_reg, base_layer]


def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):
    # conv block time distributed
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.TimeDistributed(
        layers.Convolution2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
        input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)

    x = layers.TimeDistributed(layers.BatchNormalization(axis=3), name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(
        layers.Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable,
                             kernel_initializer='normal'), name=conv_name_base + '2b')(x)

    x = layers.TimeDistributed(layers.BatchNormalization(axis=3), name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(layers.Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'),
                               name=conv_name_base + '2c', trainable=trainable)(x)

    x = layers.TimeDistributed(layers.BatchNormalization(axis=3), name=bn_name_base + '2c')(x)

    shortcut = layers.TimeDistributed(
        layers.Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'),
        name=conv_name_base + '1')(input_tensor)

    shortcut = layers.TimeDistributed(layers.BatchNormalization(axis=3), name=bn_name_base + '1')(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):
    # identity block time distributed
    nb_filter1, nb_filter2, nb_filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.TimeDistributed(
        layers.Convolution2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'),
        name=conv_name_base + '2a')(input_tensor)

    x = layers.TimeDistributed(layers.BatchNormalization(axis=3), name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(
        layers.Convolution2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',
                             padding='same'), name=conv_name_base + '2b')(x)

    x = layers.TimeDistributed(layers.BatchNormalization(axis=3), name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(
        layers.Convolution2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'),
        name=conv_name_base + '2c')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(axis=3), name=bn_name_base + '2c')(x)

    x = layers.Add()([x, input_tensor])
    x = layers.Activation('relu')(x)

    return x


def classifier_layers(x, input_shape, trainable=False):
    x = conv_block_td(x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='a', input_shape=input_shape,
                      strides=(2, 2), trainable=trainable)
    x = identity_block_td(x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='b', trainable=trainable)
    x = identity_block_td(x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='c', trainable=trainable)
    x = layers.TimeDistributed(layers.AveragePooling2D((7, 7)), name='avg_pool')(x)

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
    input_shape = (num_rois, 14, 14, 1024)

    x = [base_layers, input_rois]
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)(x)
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    out = layers.TimeDistributed(layers.Flatten())(out)
    out_class = layers.TimeDistributed(layers.Dense(nb_classes, activation='softmax', kernel_initializer='zero'),
                                       name='dense_class_{}'.format(nb_classes))(out)
    out_regr = layers.TimeDistributed(
        layers.Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
        name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]
