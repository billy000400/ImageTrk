import numpy as np
from matplotlib import pyplot as plt

def plot_hits_and_windwos(hits_all, windows):
    hitsInTracks = []
    table = []
    for trkIdx, hitIdcPdgId in track_all.items():
        hitIdc = hitIdcPdgId[:-1]
        hitsPerTrack = [hit_all[idx] for idx in hitIdc]
        tsPerTrack = [hit[3] for hit in hitsPerTrack]
        delta_t= max(tsPerTrack)-min(tsPerTrack)
        if delta_t > 1000:
            continue
        hitsInTracks.append(hitsPerTrack)

    hits = [hit for hitsPerTrack in hitsInTracks for hit in hitsPerTrack]
    zs = [hit[2] for hit in hits]
    ts = [hit[3] for hit in hits]
    return
