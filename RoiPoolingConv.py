from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf


class RoiPoolingConv(Layer):
    """
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img: (1, rows, cols, channels)
        X_roi:(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape: `(1, num_rois, channels, pool_size, pool_size)`
    """

    def __init__(self, pool_size, num_rois, **kwargs):
        self.nb_channels = None
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert (len(x) == 2), 'X must be list [X_img, X_roi]'
        img = x[0]
        rois = x[1]
        outputs = []

        for roi_idx in range(self.num_rois):
            x = K.cast(rois[0][roi_idx][0], 'int32')
            y = K.cast(rois[0][roi_idx][1], 'int32')
            w = K.cast(rois[0][roi_idx][2], 'int32')
            h = K.cast(rois[0][roi_idx][3], 'int32')

            rs = tf.image.resize(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))
        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size, 'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dir(list(base_config.items()) + list(config.items()))


