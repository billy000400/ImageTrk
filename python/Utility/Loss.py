# Loss functions for training the Mu2e Faster R-CNN system
import sys
from pathlib import Path
import random

import numpy as np
import numpy.ma as ma

import tensorflow as tf
from tensorflow.math import log
import tensorflow.keras.backend as K
from tensorflow.keras.losses import (
    BinaryCrossentropy,
    CategoricalCrossentropy,
    Huber,
    MeanSquaredError,
    Reduction
)

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from Information import*


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

def define_detector_class_loss(reg):

    def class_loss(cls_r, cls_p):

        cce = CategoricalCrossentropy(reduction=Reduction.SUM)
        score = cce(cls_r, cls_p)
        N = tf.size(cls_r)
        N = tf.cast(N, tf.float32)
        loss =  score/N*reg

        return loss

    return class_loss

def define_detector_regr_loss(reg):

    def regr_loss(bbox_r, bbox_p):
        mask = ~tf.math.is_nan(bbox_r)

        mb_r = tf.boolean_mask(bbox_r, mask=mask)
        mb_p = tf.boolean_mask(bbox_p, mask=mask)

        h = Huber(reduction=Reduction.SUM)
        score = h(mb_r, mb_p)

        N_reg = tf.size(mb_r)
        N_reg = tf.cast(N_reg, tf.float32)

        return score/N_reg*reg
        return loss

    return regr_loss

def define_detector_loss(regs):
    reg_1, reg_2 = regs

    loss_class = define_detector_class_loss(reg_1)
    loss_regr = define_detector_regr_loss(reg_2)

    def detector_loss(y_real, y_predict):
        cls_r, box_r = y_real
        cls_p, box_p = y_predicts

        loss_class_val = loss_class(cls_r, cls_p)
        loss_regr_val = loss_regr(cls_r, box_r, box_p)

        return loss_class_val+loss_regr_val

    return detector_loss

def unmasked_cce(y_real, y_predict):
    # sl = simple layer = the 1st layer
    y_real_sl = y_real[:,:,:,0]
    mask = ~tf.math.is_nan(y_real_sl)

    # my: masked y
    my_real = tf.boolean_mask(y_real, mask)
    my_predict = tf.boolean_mask(y_predict, mask)
    cce = CategoricalCrossentropy(reduction=Reduction.SUM)
    score = cce(my_real, my_predict)
    N = tf.size(my_real)
    N = tf.cast(N, tf.float32)
    # tf.keras.backend.print_tensor(y_real_sl)
    return score/N

def weighted_cce(y_real, y_predict):

    major_mask = y_real[:,:,:,2]==1
    bg_mask = y_real[:,:,:,1]==1
    blank_mask = y_real[:,:,:,0]==1

    major_indices = tf.where(major_mask)
    bg_indices = tf.where(bg_mask)
    blank_indices = tf.where(blank_mask)

    majorNum = tf.size(major_indices)
    bgNum = tf.size(bg_indices)
    blankNum = tf.size(blank_indices)

    y_real_major = tf.boolean_mask(y_real, major_mask)
    y_predict_major = tf.boolean_mask(y_predict, major_mask)

    y_real_bg = tf.boolean_mask(y_real, bg_mask)
    y_predict_bg = tf.boolean_mask(y_predict, bg_mask)

    y_real_blank = tf.boolean_mask(y_real, blank_mask)
    y_predict_blank = tf.boolean_mask(y_predict, blank_mask)

    numArr = [majorNum, bgNum, blankNum]
    sum = majorNum+bgNum+blankNum
    numArr = tf.where(tf.equal(numArr,0), sum, numArr)
    weights = sum/numArr
    weights = tf.cast(weights, tf.float32)
    # tf.keras.backend.print_tensor(weights)

    cce = CategoricalCrossentropy(reduction=Reduction.SUM)

    score_major = cce(y_real_major, y_predict_major) *  weights[0]
    score_bg = cce(y_real_bg, y_predict_bg) * weights[1]
    score_blank = cce(y_real_blank, y_predict_blank) * weights[2]
    score = score_major+score_bg+score_blank
    N = tf.size(y_real[0])
    N = tf.cast(N, tf.float32)

    return score/N

def top2_weighted_cce(y_real, y_predict):

    major_mask = y_real[:,:,:,2]==1
    bg_mask = y_real[:,:,:,1]==1

    major_indices = tf.where(major_mask)
    bg_indices = tf.where(bg_mask)

    majorNum = tf.cast(tf.size(major_indices), tf.float32)
    bgNum = tf.cast(tf.size(bg_indices), tf.float32)

    numArr = [majorNum, bgNum]
    sum = majorNum+bgNum
    numArr = tf.where(tf.equal(numArr,0), sum, numArr)
    weights = tf.reduce_min(numArr)/numArr
    weights = tf.cast(weights, tf.float32)

    y_real_major = tf.boolean_mask(y_real, major_mask)
    y_predict_major = tf.boolean_mask(y_predict, major_mask)

    y_real_bg = tf.boolean_mask(y_real, bg_mask)
    y_predict_bg = tf.boolean_mask(y_predict, bg_mask)

    cce = CategoricalCrossentropy(reduction=Reduction.SUM)

    score_major = cce(y_real_major, y_predict_major) *  weights[0]
    score_bg = cce(y_real_bg, y_predict_bg) * weights[1]

    score = (score_major+score_bg)/(majorNum+bgNum)

    return score

def WeightedCCE(Y):
    num_class = Y.shape[-1]

    return

def categorical_focal_loss(alpha, gamma=2.):

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_real, y_predict):

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = 1e-6
        y_predict = tf.clip_by_value(y_predict, epsilon, 1. - epsilon)

        major_mask = y_real[:,:,:,2]==1
        bg_mask = y_real[:,:,:,1]==1
        blank_mask = y_real[:,:,:,0]==1

        y_real_major = tf.boolean_mask(y_real, major_mask)
        y_predict_major = tf.boolean_mask(y_predict, major_mask)

        y_real_bg = tf.boolean_mask(y_real, bg_mask)
        y_predict_bg = tf.boolean_mask(y_predict, bg_mask)

        y_real_blank = tf.boolean_mask(y_real, blank_mask)
        y_predict_blank = tf.boolean_mask(y_predict, blank_mask)

        cce = CategoricalCrossentropy(reduction=Reduction.NONE)

        score_major = cce(y_real_major, y_predict_major)
        # x shape is different from the y shape
        score_major = score_major * tf.math.pow(1-y_predict_major[:,2], gamma)
        score_major = score_major *  alpha[0]
        score_major = tf.math.reduce_sum(score_major)

        score_bg = cce(y_real_bg, y_predict_bg)
        score_bg = score_bg * tf.math.pow(1-y_predict_bg[:,1], gamma)
        score_bg = score_bg * alpha[1]
        score_bg = tf.math.reduce_sum(score_bg)

        score_blank = cce(y_real_blank, y_predict_blank)
        scoire_blank = score_blank * tf.math.pow(1-y_predict_blank[:,0], gamma)
        score_blank = score_blank * alpha[2]
        score_blank = tf.math.reduce_sum(score_blank)

        score = score_major+score_bg+score_blank
        N = tf.size(y_real[0])
        N = tf.cast(N, tf.float32)

        return score/N

    return categorical_focal_loss_fixed
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
