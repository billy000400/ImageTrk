# import begins
import sys
from pathlib import Path
import timeit
import csv

import numpy as np
import pandas as pd

from sqlalchemy import *

util_dir = Path.cwd().parent.joinpath('util')
sys.path.insert(1, str(util_dir))
from TrackDB_Classes import *
from mu2e_output import *
### import ends

### configure parameters
hitsNumCut = 9
windowNum = 400
mean = 10
std = 3
np.random.seed(0)

### configure io
track_str = '/home/Billy/Mu2e/analysis/data/tracks'
db_name = 'dig.mu2e.CeEndpoint.MDC2018b.001002_00000172.art'
db_file = Path(track_str).joinpath(db_name+'.db')

tracks = []
data_dir = Path.cwd().parent.parent.joinpath('data')
reference_file = data_dir.joinpath('tracks_reference.csv')


### make windows and assign track numebr to every window
floats = np.random.normal(loc=mean, scale=std, size=windowNum)
float_type_ints = np.around(floats)
track_numbers = float_type_ints.astype(int)
ptclNum = track_numbers.sum()

### get all particles
pinfo('Connecting to the track database')
engine = create_engine('sqlite:///'+str(db_file))
from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)
session = Session()

pinfo('Searching for major tracks')
ptcl_ids = session.query(Particle.id).all()
ptcl_ids = [ tp[0] for tp in ptcl_ids ]
ptcl_ids = [ id for id in ptcl_ids if (session.query(StrawHit).filter(StrawHit.particle==id).count()>=hitsNumCut) ]
ptcl_ids = ptcl_ids[:ptclNum]

hitNum_tot = 0
### filter out all major tracks
for idx, ptcl_id in enumerate(ptcl_ids):
    sys.stdout.write(t_info(f'Parsing particles {idx+1}/{ptclNum}', special='\r'))
    if idx+1 == ptclNum:
        sys.stdout.write('\n')
    sys.stdout.flush()

    hit_id_qrl = session.query(StrawHit.id).filter(StrawHit.particle==ptcl_id)
    hitNum_tot += hit_id_qrl.count()
    hit_ids = hit_id_qrl.all()
    hit_ids = [id for id, in hit_ids]
    tracks.append(hit_ids)

### save to local
ref_file_open = open(reference_file, 'w')
writer = csv.writer(ref_file_open)
for track in tracks:
    writer.writerow(track)
ref_file_open.close()

pinfo(f'{hitNum_tot}')
