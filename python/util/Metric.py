import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.math import exp
from tensorflow.keras.metrics import (
    binary_accuracy,
    categorical_accuracy
)
from tensorflow.keras.backend import print_tensor


def union(rec_a, rec_b, intersection):
    area_a = (rec_a[1]-rec_a[0])*(rec_a[3]-rec_a[2])
    area_b = (rec_b[1]-rec_b[0])*(rec_b[3]-rec_b[2])
    return area_a+area_b-intersection

def intersection(rec_a, rec_b):
    # rec_a(b) should be (xmin, xmax, ymin, ymax)
    w = tf.math.reduce_min([rec_a[1], rec_b[1]]) - tf.math.reduce_max([rec_a[0], rec_b[0]])
    h = tf.math.reduce_min([rec_a[3], rec_b[3]]) - tf.math.reduce_max([rec_a[2], rec_b[2]])
    def f1(): return 0.0
    def f2(): return w*h
    result = tf.cond( tf.math.logical_or( tf.less(w,0.0), tf.less(h,0.0) ), f1, f2)
    return result

def iou(rec_a, rec_b):
    overlap = intersection(rec_a, rec_b)
    sum = union(rec_a, rec_b, overlap)
    return overlap/sum


def unmasked_binary_accuracy(p_r, p_p):
    mask = ~tf.math.is_nan(p_r)

    mp_r = tf.boolean_mask(p_r, mask=mask)
    mp_p = tf.boolean_mask(p_p, mask=mask)

    score_ew = binary_accuracy(mp_r, mp_p)
    score = tf.math.reduce_sum(score_ew)

    N_cls = tf.size(score_ew)
    N_cls = tf.cast(N_cls, tf.float32)

    return score/N_cls*100.0

def unmasked_categorical_accuracy(p_r, p_p):
    p_real_sl = p_r[:,:,:,0]
    mask = ~tf.math.is_nan(p_real_sl)

    mp_r = tf.boolean_mask(p_r, mask=mask)
    mp_p = tf.boolean_mask(p_p, mask=mask)
    score_ew = categorical_accuracy(mp_r, mp_p)
    score = tf.math.reduce_sum(score_ew)

    N_cls = tf.size(score_ew)
    N_cls = tf.cast(N_cls, tf.float32)

    return score/N_cls*100.0

def top2_categorical_accuracy(y_real, y_predict):

    major_mask = y_real[:,:,:,2]==1
    bg_mask = y_real[:,:,:,1]==1

    y_real_major = tf.boolean_mask(y_real, major_mask)
    y_predict_major = tf.boolean_mask(y_predict, major_mask)

    y_real_bg = tf.boolean_mask(y_real, bg_mask)
    y_predict_bg = tf.boolean_mask(y_predict, bg_mask)

    score_major_ew = categorical_accuracy(y_real_major, y_predict_major)
    score_bg_ew = categorical_accuracy(y_real_bg, y_predict_bg)

    score_major = tf.math.reduce_sum(score_major_ew)
    score_bg = tf.math.reduce_sum(score_bg_ew)

    N_major = tf.size(score_major_ew)
    N_major = tf.cast(N_major, tf.float32)
    N_bg = tf.size(score_bg_ew)
    N_bg = tf.cast(N_bg, tf.float32)

    sum = N_major + N_bg

    return (score_major+score_bg)/sum*100

def unmasked_IoU(t_r, t_p):
    mask = ~tf.math.is_nan(t_r)

    mt_r = tf.boolean_mask(t_r, mask=mask)
    mt_p = tf.boolean_mask(t_p, mask=mask)

    mt_r_4 = tf.reshape(mt_r, [tf.size(mt_r)/4, 4])
    mt_p_4 = tf.reshape(mt_p, [tf.size(mt_p)/4, 4])

    rx = tf.gather(mt_r_4, 0, axis=1)
    ry = tf.gather(mt_r_4, 1, axis=1)
    log_rw = tf.gather(mt_r_4, 2, axis=1)
    log_rh = tf.gather(mt_r_4, 3, axis=1)
    rw = exp(log_rw)
    rh = exp(log_rh)
    rx1 = rx-rw/2
    rx2 = rx+rw/2
    ry1 = ry-rh/2
    ry2 = ry+rh/2
    rec_r = tf.stack([rx1, rx2, ry1, ry2], axis=1)

    px = tf.gather(mt_p_4, 0, axis=1)
    py = tf.gather(mt_p_4, 1, axis=1)
    log_pw = tf.gather(mt_p_4, 2, axis=1)
    log_ph = tf.gather(mt_p_4, 3, axis=1)
    pw = exp(log_pw)
    ph = exp(log_ph)
    px1 = px-pw/2
    px2 = px+pw/2
    py1 = py-ph/2
    py2 = py+ph/2
    rec_p = tf.stack([px1, px2, py1, py2], axis=1)

    rowNum = tf.shape(rec_r)[0]
    i = 0
    iou_tot = 0.0

    def add_i(i, rowNum, iou_tot):
        return [tf.add(i,1), rowNum, tf.add(iou_tot, iou(rec_r[i], rec_p[i])) ]

    def c(i, rowNum, iou_tot):
        return tf.less(i,rowNum)

    i, rowNum, iou_tot = tf.while_loop(c, add_i, [i, rowNum, iou_tot])

    rowNum = tf.cast(rowNum, tf.float32)
    return iou_tot/rowNum

def positive_number(p_r, p_p):

    positive_truth = tf.math.greater(p_p, 0.5)
    pos_num = tf.reduce_sum(tf.cast(positive_truth, tf.float32))

    batch_size = tf.shape(p_r)[0]
    batch_size = tf.cast(batch_size, tf.float32)

    mask = ~tf.math.is_nan(p_r)
    mp_r = tf.boolean_mask(p_r, mask=mask)

    # print_tensor(batch_size, 'batch_size: ')
    # print_tensor(tf.reduce_sum(mp_r)/batch_size, 'sampled positive: ')
    # print_tensor(pos_num/batch_size, 'predicted positive: ')

    return pos_num/batch_size # denominator is batch size
