import sys
from pathlib import Path
import random

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
