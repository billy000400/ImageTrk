import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.image import non_max_suppression_with_scores

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

class RoIPooling2D(Layer):
    def __init__(self):
        return

    def build(self):
        return

    def call(self, x, training=False):
        return
