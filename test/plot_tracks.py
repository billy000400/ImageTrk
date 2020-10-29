import sys
script, trkNum = sys.argv
trkNum = int(trkNum)
from pathlib import Path

import random
random.seed(3)

import matplotlib.pyplot as plt

from sqlalchemy import *

util_dir = Path.cwd().parent.joinpath('python/util')
sys.path.insert(1, str(util_dir))
from TrackDB_Classes import *

track_dir = Path("../tracks")
dp_name = "dig.mu2e.CeEndpoint.MDC2018b.001002_00000011.art"
db_file = track_dir.joinpath(dp_name+".db")
cwd = Path.cwd()

engine = create_engine('sqlite:///'+str(db_file))
# make session
from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine) # session factory
session = Session() # session object

ptcl_ids = session.query(StrawHit.particle).distinct().all()
ptcl_ids = [ x for (x,) in ptcl_ids]

found = 0
id_plot = []
while found < trkNum:
    id = random.sample(ptcl_ids, 1)[0]
    if session.query(StrawHit.particle).filter(StrawHit.particle==id).count() > 9:
        found += 1
        id_plot.append(id)
    else:
        continue

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
ax1.set(title='MC Truth', xlabel='x', ylabel='y')
ax2.set(title='Reco Signal', xlabel='x', ylabel='y')
for id in id_plot:
    hits = session.query(StrawHit).filter(StrawHit.particle==id).all()
    mcs = session.query(StrawDigiMC).filter(StrawDigiMC.particle==id).all()

    x,y,x_reco,y_reco = [ [] for i in range(4) ]
    for mc in mcs:
        x.append(mc.x)
        y.append(mc.y)
    for hit in hits:
        x_reco.append(hit.x_reco)
        y_reco.append(hit.y_reco)

    ax1.scatter(x,y,s=5)
    ax2.scatter(x_reco,y_reco,s=5)
plt.show()
