# Loss functions for training the Mu2e Faster R-CNN system
import sys
from pathlib import Path
import random

import numpy as np
import numpy.ma as ma

import tensorflow as tf
from tensorflow.math import log
from tensorflow.keras.losses import BinaryCrossentropy, Huber, MeanSquaredError, Reduction

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from mu2e_output import*


def log_loss(y, p):
    """ Compute the log loss by tensorflow from binary truth and predicted score

    Parameters
    ----------
    y:
        The binary truth: 0 or 1
    p:
        The predicted score in [0, 1]

    Returns
    -------
    the value of the log loss
    """
    return -( y*log(p) + (1-y)*log(1-p) )

def define_rpn_class_loss(reg):
    def rpn_class_loss(p_r, p_p):

        mask = ~tf.math.is_nan(p_r)

        mp_r = tf.boolean_mask(p_r, mask=mask)
        mp_p = tf.boolean_mask(p_p, mask=mask)
        #tf.keras.backend.print_tensor(tf.math.reduce_any(~tf.math.is_nan(mp_r)))
        #tf.keras.backend.print_tensor(tf.math.reduce_any(~tf.math.is_nan(mp_r)),'[mp_r]')

        bce = BinaryCrossentropy(reduction=Reduction.SUM)
        score = bce(mp_r, mp_p)
        #tf.keras.backend.print_tensor(score,'[sum score]')

        N_cls = tf.size(mp_r)
        N_cls = tf.cast(N_cls, tf.float32)
        #tf.keras.backend.print_tensor(N_cls,'[N_cls]')

        return score/N_cls*reg

    return rpn_class_loss

def define_rpn_regr_loss(reg):
    def rpn_regr_loss(t_r, t_p):
        mask = ~tf.math.is_nan(t_r)

        mt_r = tf.boolean_mask(t_r, mask=mask)
        mt_p = tf.boolean_mask(t_p, mask=mask)

        h = Huber(reduction=Reduction.SUM)
        score = h(mt_r, mt_p)

        N_reg = tf.size(mt_r)
        N_reg = tf.cast(N_reg, tf.float32)

        return score/N_reg*reg

    return rpn_regr_loss

def define_rpn_loss(regs):
    reg_1 = regs[0]
    reg_2 = regs[1]

    loss_class = define_rpn_class_loss(reg_1)
    loss_regr = define_rpn_regr_loss(reg_2)

    def rpn_loss(y_real, y_predict):
        lm_r = y_real[0]
        dm_r = y_real[1]

        lm_p = y_predict[0]
        dm_p = y_predict[1]

        loss_class_val = loss_class(lm_r, lm_p)
        loss_regr_val = loss_regr(dm_r, dm_p)

        return loss_class_val+loss_regr_val

    return rpn_loss

# test benches
def test_rpn_class_loss():
    # test tensorship is (1,3,3,1)
    test_r = np.random.rand(1,18,18,84)
    test_p = np.random.rand(1,18,18,84)

    test_r = tf.convert_to_tensor(test_r,dtype=tf.float32)
    test_p = tf.convert_to_tensor(test_p,dtype=tf.float32)

    rpn_class_loss = define_rpn_class_loss(1)
    loss = rpn_class_loss(test_r,test_p)
    pdebug(loss)

def test_data():
    tensor = [[1, 2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9, 10], [5, 6, 7, 8, 9, 10, 11, 12]]
    mask = np.array([[True, True, True, True, False, False, False, False],\
                    [False, False, False, False, True, True, True, True],\
                    [True, True, True, True, False, False, False, False]])
    result = tf.boolean_mask(tensor,mask)
    size = tf.size(result)
    pdebug(result)
    result = tf.reshape(result, [size/4, 4])
    pdebug(result)
    pdebug(result[:,0])
