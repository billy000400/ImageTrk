
def union(rec_a, rec_b, intersection):
    area_a = (rec_a[1]-rec_a[0])*(rec_a[3]-rec_a[2])
    area_b = (rec_b[1]-rec_b[0])*(rec_b[3]-rec_b[2])
    return area_a+area_b-intersection

def intersection(rec_a, rec_b):
    # rec_a(b) should be (xmin, xmax, ymin, ymax)
    w = min(rec_a[1], rec_b[1]) - max(rec_a[0], rec_b[0])
    h = min(rec_a[3], rec_b[3]) - max(rec_a[2], rec_b[2])
    if w<0 or h<0:
        return 0
    return w*h

def iou(rec_a, rec_b):
    overlap = intersection(rec_a, rec_b)
    sum = union(rec_a, rec_b, overlap)
    return overlap/sum
