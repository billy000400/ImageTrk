import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.math import exp
from tensorflow.keras.metrics import (
    binary_accuracy,
    categorical_accuracy,
    Precision
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

def union1D(intv_a, intv_b, intersection):
    w_a = intv_a[1]-intv_a[0]
    w_b = intv_b[1]-intv_b[0]
    return w_a+w_b-intersection

def intersection1D(intv_a, intv_b):
    # rec_a(b) should be (xmin, xmax, ymin, ymax)
    w = tf.math.reduce_min([intv_a[1], intv_b[1]]) - tf.math.reduce_max([intv_a[0], intv_b[0]])
    def f1(): return 0.0
    def f2(): return w
    result = tf.cond(tf.less(w,0.0), f1, f2)
    return result

def iou1D(intv_a, intv_b):
    overlap = intersection1D(intv_a, intv_b)
    sum = union1D(intv_a, intv_b, overlap)
    return overlap/sum

def unmasked_binary_accuracy(p_r, p_p):
    mask = ~tf.math.is_nan(p_r)
    #mask.set_shape([None,32,32,18])
    mp_r = tf.boolean_mask(p_r, mask=mask)
    mp_p = tf.boolean_mask(p_p, mask=mask)

    score_ew = binary_accuracy(mp_r, mp_p)
    score = tf.math.reduce_sum(score_ew)

    N_cls = tf.size(score_ew)
    N_cls = tf.cast(N_cls, tf.float32)

    return score/N_cls*100.0

def weighted_unmasked_binary_accuracy(y_real, y_predict):

    major_mask = (y_real==1)
    bg_mask = (y_real==0)

    y_real_major = tf.boolean_mask(y_real, major_mask)
    y_predict_major = tf.boolean_mask(y_predict, major_mask)

    y_real_bg = tf.boolean_mask(y_real, bg_mask)
    y_predict_bg = tf.boolean_mask(y_predict, bg_mask)


    score_major_avg = binary_accuracy(y_real_major, y_predict_major)
    score_bg_avg = binary_accuracy(y_real_bg, y_predict_bg)

    N_major = tf.size(y_real_major)
    N_major = tf.cast(N_major, tf.float32)
    N_bg = tf.size(y_real_bg)
    N_bg = tf.cast(N_bg, tf.float32)

    sum = N_major + N_bg

    return (score_major_avg*N_bg+score_bg_avg*N_major)/sum*100


def unmasked_precision(p_r, p_p):
    mask = ~tf.math.is_nan(p_r)
    #mask.set_shape([None,32,32,18])
    mp_r = tf.boolean_mask(p_r, mask=mask)
    mp_p = tf.boolean_mask(p_p, mask=mask)

    pred_positive_mask = (mp_p>=0.5)
    pred_positive = tf.math.reduce_sum(tf.cast(pred_positive_mask, tf.float32))
    mmp_r = tf.boolean_mask(mp_r, mask=pred_positive_mask)
    true_positive = tf.math.reduce_sum(tf.cast(mmp_r==1, tf.float32))

    precision = true_positive/pred_positive*100
    return tf.where(tf.math.is_nan(precision), 0.0, precision) 

def unmasked_recall(p_r, p_p):
    mask = ~tf.math.is_nan(p_r)
    #mask.set_shape([None,32,32,18])
    mp_r = tf.boolean_mask(p_r, mask=mask)
    mp_p = tf.boolean_mask(p_p, mask=mask)

    tot_positive_mask = (mp_r==1)
    tot_positive = tf.math.reduce_sum(tf.cast(tot_positive_mask,tf.float32))
    mmp_p = tf.boolean_mask(mp_p, mask=tot_positive_mask)
    true_positive = tf.math.reduce_sum(tf.cast(mmp_p>=0.5,tf.float32))

    return true_positive/tot_positive*100.0

def unmasked_categorical_accuracy(p_r, p_p):
    # p_real_sl = p_r[:,:,:,0]
    # mask = ~tf.math.is_nan(p_real_sl)
    mask = ~tf.math.is_nan(p_r)

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
    #mask.set_shape([None,32,32,72])
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

def unmasked_IoUV2(t_r, t_p):
    # IoU for Fast R-CNN
    mask = ~tf.math.is_nan(t_r)

    mt_r = tf.boolean_mask(t_r, mask=mask)
    mt_p = tf.boolean_mask(t_p, mask=mask)

    mt_r_4 = tf.reshape(mt_r, [tf.size(mt_r)/4, 4])
    mt_p_4 = tf.reshape(mt_p, [tf.size(mt_p)/4, 4])

    rx = tf.gather(mt_r_4, 0, axis=1)
    ry = tf.gather(mt_r_4, 1, axis=1)
    rw = tf.gather(mt_r_4, 2, axis=1)
    rh = tf.gather(mt_r_4, 3, axis=1)

    rx1 = rx
    rx2 = rx+rw
    ry1 = ry-rh
    ry2 = ry
    rec_r = tf.stack([rx1, rx2, ry1, ry2], axis=1)

    px = tf.gather(mt_p_4, 0, axis=1)
    py = tf.gather(mt_p_4, 1, axis=1)
    pw = tf.gather(mt_p_4, 2, axis=1)
    ph = tf.gather(mt_p_4, 3, axis=1)
    px1 = px
    px2 = px+pw
    py1 = py-ph
    py2 = py
    rec_p = tf.stack([px1, px2, py1, py2], axis=1)

    rowNum = tf.shape(rec_r)[0]
    i = 0
    iou_tot = 0.0

    def add_i(i, rowNum, iou_tot):
        iou_val = iou(rec_r[i], rec_p[i])
        return [tf.add(i,1), rowNum, tf.add(iou_tot, iou_val) ]

    def c(i, rowNum, iou_tot):
        return tf.less(i,rowNum)

    i, rowNum, iou_tot = tf.while_loop(c, add_i, [i, rowNum, iou_tot])

    rowNum = tf.cast(rowNum, tf.float32)
    return iou_tot/rowNum

def unmasked_IoU1D(t_r, t_p):

    mask = ~tf.math.is_nan(t_r)

    mt_r = tf.boolean_mask(t_r, mask=mask) # shape = (batch, 256, 1, 2)
    mt_p = tf.boolean_mask(t_p, mask=mask)

    mt_r_2 = tf.reshape(mt_r, [tf.size(mt_r)/2, 2])
    mt_p_2 = tf.reshape(mt_p, [tf.size(mt_p)/2, 2])

    rc = tf.gather(mt_r_2, 0, axis=1)
    log_rw = tf.gather(mt_p_2, 1, axis=1)
    rw = exp(log_rw)
    rt1 = rc-rw/2
    rt2 = rc+rw/2
    intv_r = tf.stack([rt1, rt2], axis=1)

    pc = tf.gather(mt_p_2, 0, axis=1)
    log_pw = tf.gather(mt_p_2, 1, axis=1)
    pw = exp(log_pw)
    pt1 = pc-pw/2
    pt2 = pc+pw/2
    intv_p = tf.stack([pt1, pt2], axis=1)

    rowNum = tf.shape(intv_r)[0]
    i = 0
    iou_tot = 0.0

    def add_i(i, rowNum, iou_tot):
        return [tf.add(i,1), rowNum, tf.add(iou_tot, iou1D(intv_r[i], intv_p[i])) ]

    def c(i, rowNum, iou_tot):
        return tf.less(i,rowNum)

    i, rowNum, iou_tot = tf.while_loop(c, add_i, [i, rowNum, iou_tot])

    rowNum = tf.cast(rowNum, tf.float32)
    return iou_tot/rowNum

# hit purity: real major/predicted major
def hit_purity(y_r, y_p):
    predict_major_mask = tf.argmax(y_p, axis=3, output_type=tf.int32)==2

    y_predict_major = tf.boolean_mask(y_p, predict_major_mask)
    y_real_selected = tf.boolean_mask(y_r, predict_major_mask)

    binary_purity = categorical_accuracy(y_real_selected, y_predict_major)
    binary_sum = tf.math.reduce_sum(binary_purity)

    N = tf.cast(tf.size(binary_purity), tf.float32)
    N = tf.math.maximum(N,1.0)

    purity = binary_sum/N*100

    return purity

# hit efficiency: real major/all major
def hit_efficiency(y_r, y_p):
    real_major_mask = y_r[:,:,:,2]==1

    y_real_major = tf.boolean_mask(y_r, real_major_mask)
    y_predict_selected = tf.boolean_mask(y_p, real_major_mask)

    binary_purity = categorical_accuracy(y_real_major, y_predict_selected)
    binary_sum = tf.math.reduce_sum(binary_purity)

    N = tf.cast(tf.size(binary_purity), tf.float32)
    N = tf.math.maximum(N,1.0)

    efficiency = binary_sum/N*100

    return efficiency

def positive_number(p_r, p_p):

    positive_truth = tf.math.greater(p_p, 0.5)
    pos_num = tf.reduce_sum(tf.cast(positive_truth, tf.float32))

    batch_size = tf.shape(p_r)[0]
    batch_size = tf.cast(batch_size, tf.float32)

    mask = ~tf.math.is_nan(p_r)
    mp_r = tf.boolean_mask(p_r, mask=mask)

    # print_tensor(tf.reduce_sum(tf.cast(tf.math.greater(p_r,0.5), tf.float32)), message='pos')
    # print_tensor(tf.reduce_sum(tf.cast(tf.math.less(p_r,0.5), tf.float32)), message='neg')
    # print_tensor(batch_size, 'batch_size: ')
    # print_tensor(tf.reduce_sum(mp_r)/batch_size, 'sampled positive: ')
    # print_tensor(pos_num, 'total positive number: ')

    return pos_num/batch_size # denominator is batch size
