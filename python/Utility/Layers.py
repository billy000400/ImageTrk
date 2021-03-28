import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D
from tensorflow.image import non_max_suppression_with_scores
from tensorflow.keras.initializers import RandomNormal

initializer = RandomNormal(stddev=0.01)

class rpn():
    def __init__(self, anchor_scales, anchor_ratios):
        self.num_anchors = len(anchor_scales) * len(anchor_ratios)
        self.shared_layer = Conv2D(512, (3, 3), padding='same', activation = 'relu', kernel_initializer=initializer, name='rpn_conv1')

    def classifier(self, base):
        x = self.shared_layer(base)
        x_class = Conv2D(self.num_anchors, (1, 1), activation='sigmoid', kernel_initializer=initializer, name='rpn_out_class')(x)
        return x_class

    def regression(self, base):
        x = self.shared_layer(base)
        x_regr = Conv2D(self.num_anchors * 4, (1, 1), activation='linear', kernel_initializer=initializer, name='rpn_out_regress')(x)
        return x_regr

### !!! Not Working
### A Tensorflow.keras layer propose bboxes according to the RPN outputs
# A Non-Maximum Suppression is embedded.
class RPN_to_RoI(Layer):
    def __init__(self, anchors, max_output_size=100, iou_threshold=0.9,\
        score_threshold=0.9, soft_nms_sigma=0.0):
        self.anchors = anchors
        self.mos = max_output_size
        self.it = iou_threshold
        self.st = score_threshold
        self.sigma = soft_nms_sigma
        super().__init__()

    # A method modified from Abstract.translate_delta
    def translate_delta(self, anchor, delta):
        tx, ty, tw, th = delta
        xa = (anchor[0]+anchor[1])/2
        ya = (anchor[2]+anchor[3])/2
        wa = anchor[1]-anchor[0]
        ha = anchor[3]-anchor[2]

        x = tx*wa+xa
        y = ty*ha+ya
        w = tf.math.exp(tw)*wa
        h = tf.math.exp(th)*ha

        xmin = x-w/2
        xmax = x+w/2
        ymin = y-h/2
        ymax = y+h/2

        result = [xmin, xmax, ymin, ymax]
        return result


    def call(self, x):


        def propose_score_bbox_pair(mini_batch):

            score_map, delta_map = mini_batch
            anchors = self.anchors

            scores = []
            bbox_raws = []

            threshold = 0.5

            for i, j, k in np.ndindex((score_map.shape)):
                score = score_map[i,j,k]
                if score > threshold:
                    anchor = anchors[i][j][k]
                    delta = delta_map[i,j,4*k:4*k+4]
                    bbox = translate_delta(anchor, delta)
                    scores.append(score)
                    bbox_raws.append(bbox)

            # trim bboxes
            for i, bbox in enumerate(bbox_raws):
                if bbox[0] < 0:
                    bbox_raws[i][0] = 0
                if bbox[1] > 1:
                    bbox_raws[i][1] = 1
                if bbox[2] < 0:
                    bbox_raws[i][2] = 0
                if bbox[3] > 1:
                    bbox_raws[i][3] = 1

            return [scores, bbox_raws]


        score_bbox_pair_batches = tf.map_fn(fn=propose_score_bbox_pair, elems=x)

        def nms(mini_batch):

            scores, bboxes_raw = mini_batch

            scores_tf = tf.constant(scores, dtype=tf.float32)
            bboxes_raw_tf = [ [ymax, xmin, ymin, xmax] for [xmin, xmax, ymin, ymax] in bboxes_raw ]
            bboxes_raw_tf = tf.constant(bboxes_raw_tf, dtype=tf.float32)

            selected_indices, selected_score =\
                non_max_suppression_with_scores(bboxes_raw_tf, scores_tf,\
                        max_output_size=self.mos,\
                        iou_threshold=self.it, score_threshold=self.st,\
                        soft_nms_sigma=self.sigma)

            bboxes_tf = bboxes_raw_tf[selected_indices]
            return bboxes_tf

        proposal_batches = tf.map_fn(fn=nms, elems=score_bbox_pair_batches)

        return proposal_batches

class RoIPooling(Layer):
    """ Implements Region Of Interest Max Pooling
    for channel-first images and relative bounding box coordinates

    # Constructor parameters
        pooled_height, pooled_width (int) --
          specify height and width of layer outputs

    Shape of inputs
        [(batch_size, pooled_height, pooled_width, n_channels),
         (batch_size, num_rois, 4)]

    Shape of output
        (batch_size, num_rois, pooled_height, pooled_width, n_channels)
    """
    def __init__(self, pooled_height, pooled_width, **kwargs):

        self.pooled_height = pooled_height
        self.pooled_width = pooled_width

        super(RoIPooling, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """ Returns the shape of the RoI layer output
        """
        feature_map_shape, rois_shape = input_shape
        assert feature_map_shape[0] == rois_shape[0]
        batch_size = feature_map_shape[0]
        n_rois = rois_shape[1]
        n_channels = feature_map_shape[3]
        return (batch_size, n_rois, self.pooled_height,
                self.pooled_width, n_channels)

    def call(self, x):
        """ Maps the input tensor of the RoI layer to its output
                # Parameters
                    x[0] -- Convolutional feature map tensor,
                            shape (batch_size, pooled_height, pooled_width, n_channels)
                    x[1] -- Tensor of region of interests from candidate bounding boxes,
                            shape (batch_size, num_rois, 4)
                            Each region of interest is defined by four relative
                            coordinates (x_min, y_max, width, height) between 0 and 1
                # Output
                    pooled_areas -- Tensoir with the poooled region of interest, shape
                        (batch_size, num_rois, pooled_height, pooled_width, n_channels)
        """
        def curried_pool_rois(x):
            return RoIPooling._pool_rois(x[0], x[1],
                                                self.pooled_height,
                                                self.pooled_width)

        pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)

        return pooled_areas

    @staticmethod
    def _pool_rois(feature_map, rois, pooled_height, pooled_width):
        """ Applies ROI pooling for a single imgae and varios RoIs
        """
        def curried_pool_roi(roi):
            return RoIPooling._pool_roi(feature_map, roi,
                                                pooled_height, pooled_width)
        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)
        return pooled_areas

    @staticmethod
    def _pool_roi(feature_map, roi, pooled_height, pooled_width):
        """ Applies ROI Pooling to a single image and a sinmgle RoI
        """

        # Compute the region of interest
        feature_map_height = int(feature_map.shape[0])
        feature_map_width = int(feature_map.shape[1])

        # change from (x, y, w, h)
        # to (xmin, xmax, ymin, ymax)
        x = roi[0]
        y = roi[1]
        w = roi[2]
        h = roi[3]
        bbox = (x, x+w, y-h, y)

        w_start = tf.cast(feature_map_width*bbox[0], 'int32')
        w_end = tf.cast(feature_map_width*bbox[1], 'int32')
        h_start = tf.cast(feature_map_height*(1-bbox[3]), 'int32')
        h_end = tf.cast(feature_map_height*(1-bbox[2]), 'int32')

        region = feature_map[h_start:h_end, w_start:w_end, :]

        # Divide the region into non overlapping areas
        region_height = h_end - h_start
        region_width  = w_end - w_start
        h_step = tf.cast( region_height / pooled_height, 'int32')
        w_step = tf.cast( region_width  / pooled_width , 'int32')

        areas = [[(
                    i*h_step,
                    j*w_step,
                    (i+1)*h_step if i+1 < pooled_height else region_height,
                    (j+1)*w_step if j+1 < pooled_width else region_width
                   )
                   for j in range(pooled_width)]
                  for i in range(pooled_height)]

        # take the maximum of each area and stack the result
        def pool_area(x):
          return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])

        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])

        return pooled_features
