def rpn_to_RoI(anchors, score_map, delta_map):

    # select all bbox whose objective score is >= 0.5 and put them in a list
    score_bbox_pairs = propose_score_bbox_list(anchors, score_map, delta_map)

    scores, bboxes_raw = [], []
    for score_bbox in score_bbox_pairs:
        scores.append(score_bbox[0])
        bboxes_raw.append(score_bbox[1])

    # trimming bboxes
    for i, bbox in enumerate(bboxes_raw):
        if bbox[0] < 0:
            bboxes_raw[i][0] = 0
        if bbox[1] > 1:
            bboxes_raw[i][1] = 1
        if bbox[2] < 0:
            bboxes_raw[i][2] = 0
        if bbox[3] > 1:
            bboxes_raw[i][3] = 1

    scores_tf = tf.constant(scores, dtype=tf.float32)
    bboxes_raw_tf = [ [ymax, xmin, ymin, xmax] for [xmin, xmax, ymin, ymax] in bboxes_raw ]
    bboxes_raw_tf = tf.constant(bboxes_raw_tf, dtype=tf.float32)
    selected_indices, selected_scores =\
        non_max_suppression_with_scores(bboxes_raw_tf, scores_tf,\
                max_output_size=100,\
                iou_threshold=0.9, score_threshold=0.9,\
                soft_nms_sigma=0.0)

    selected_indices_list = selected_indices.numpy().tolist()
    bboxes = [ bboxes_raw[index] for index in selected_indices_list ]

    return bboxes
