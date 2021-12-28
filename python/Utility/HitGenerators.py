# @Author: Billy Li <billyli>
# @Date:   11-03-2021
# @Email:  li000400@umn.edu
# @Last modified by:   billyli
# @Last modified time: 12-28-2021



import sys
from pathlib import Path
import pickle
from collections import Counter

import numpy as np

from sqlalchemy import *
from sqlalchemy.orm import sessionmaker

util_dir = Path.cwd().parent.joinpath('Utility')
sys.path.insert(1, str(util_dir))
from Database_new import *
from Information import *

class Stochastic:
    def __init__(self, dist, db_files, hitNumCut=20):
        self.dist = dist
        self.dbs = db_files
        self.hitNumCut = hitNumCut

        self.db_iter = iter(db_files)

        self.current_db = None
        self.session = None
        self.__update_db()

        self.ptcls = None
        self.ptcl_iter = None
        self.__make_ptcl_iter()

    def __connect_db(self):
        engine = create_engine('sqlite:///'+str(self.current_db))
        Session = sessionmaker(bind=engine) # session factory
        session = Session() # session object
        self.session = session

    def __update_db(self):
        self.current_db = next(self.db_iter)
        self.__connect_db()

    def __find_ptcls(self):
        ptcls = self.session.query(Particle).all()
        self.ptcls = ptcls

    def __make_ptcl_iter(self):
        self.__find_ptcls()
        self.ptcl_iter = iter(self.ptcls)

    def generate(self, mode='eval'):
        trackNum = int(self.dist.rvs(size=1))
        trackFoundNum = 0
        tracks = {}
        hits = {}

        while trackFoundNum < trackNum:
            try:
                ptcl = next(self.ptcl_iter)
            except:
                sys.stdout.write('\n')
                sys.stdout.flush()
                pinfo('Run out of particles')
                pinfo('Connecting to the next track database')
                self.__update_db()
                self.__make_ptcl_iter()
                ptcl = next(self.ptcl_iter)

            strawHit_qrl = self.session.query(StrawDigiMC).filter(StrawDigiMC.particle==ptcl.id)
            hitNum = strawHit_qrl.count()

            if (hitNum >= self.hitNumCut) and (ptcl.pdgId == 11):
                tracks[ptcl.id] = []
                track = tracks[ptcl.id]

                strawHits = strawHit_qrl.all()
                for hit in strawHits:
                    track.append(hit.id)
                    hits[hit.id] = (hit.x, hit.y, hit.z, hit.t)

                pdgId = self.session.query(Particle.pdgId).filter(Particle.id==ptcl.id).one_or_none()[0]
                track.append(pdgId)
                trackFoundNum += 1
            else:
                continue
        if mode == 'eval':
            return hits, tracks
        else:
            return hits

class Stochastic_reco:
    def __init__(self, dist, db_files, hitNumCut=20):
        self.dist = dist
        self.dbs = db_files
        self.hitNumCut = hitNumCut

        self.db_iter = iter(db_files)

        self.current_db = None
        self.session = None
        self.__update_db()

        self.ptcls = None
        self.ptcl_iter = None
        self.__make_ptcl_iter()

    def __connect_db(self):
        engine = create_engine('sqlite:///'+str(self.current_db))
        Session = sessionmaker(bind=engine) # session factory
        session = Session() # session object
        self.session = session

    def __update_db(self):
        self.current_db = next(self.db_iter)
        self.__connect_db()

    def __find_ptcls(self):
        ptcls = self.session.query(Particle).all()
        self.ptcls = ptcls

    def __make_ptcl_iter(self):
        self.__find_ptcls()
        self.ptcl_iter = iter(self.ptcls)

    def generate(self, mode='eval'):
        trackNum = int(self.dist.rvs(size=1))
        trackFoundNum = 0
        tracks = {}
        hits = {}

        while trackFoundNum < trackNum:
            try:
                ptcl = next(self.ptcl_iter)
            except:
                sys.stdout.write('\n')
                sys.stdout.flush()
                pinfo('Run out of particles')
                pinfo('Connecting to the next track database')
                self.__update_db()
                self.__make_ptcl_iter()
                ptcl = next(self.ptcl_iter)

            strawHit_qrl = self.session.query(StrawHit).filter(StrawHit.particle==ptcl.id)
            hitNum = strawHit_qrl.count()

            if (hitNum >= self.hitNumCut) and (ptcl.pdgId == 11):
                tracks[ptcl.id] = []
                track = tracks[ptcl.id]

                strawHits = strawHit_qrl.all()
                for hit in strawHits:
                    track.append(hit.id)
                    hits[hit.id] = (hit.x_reco, hit.y_reco, hit.z_reco, hit.t_reco)

                pdgId = self.session.query(Particle.pdgId).filter(Particle.id==ptcl.id).one_or_none()[0]
                track.append(pdgId)
                trackFoundNum += 1
            else:
                continue
        if mode == 'eval':
            return hits, tracks
        else:
            return hits

class Event:
    def __init__(self, db_files, hitNumCut=0):
        self.dbs = db_files
        self.hitNumCut = hitNumCut

        self.db_iter = iter(db_files)

        self.current_db = None
        self.session = None
        self.__update_db()

        self.events = []
        self.event_iter = None
        self.__make_event_iter()

    def __connect_db(self):
        engine = create_engine('sqlite:///'+str(self.current_db))
        Session = sessionmaker(bind=engine) # session factory
        session = Session() # session object
        self.session = session

    def __update_db(self):
        self.current_db = next(self.db_iter)
        self.__connect_db()

    def __find_events(self):
        runs = self.session.query(Particle.run).distinct().all()
        for run in runs:
            run = run[0]
            ptcl_subset = self.session.query(Particle).filter(Particle.run==run).all()
            subRuns = {ptcl.subRun for ptcl in ptcl_subset}
            for subRun in subRuns:
                ptcl_subset = self.session.query(Particle).filter(Particle.run==run).\
                    filter(Particle.subRun==subRun).all()
                events = {ptcl.event for ptcl in ptcl_subset}
                for event in events:
                    self.events.append((run, subRun, event))
        return

    def __make_event_iter(self):
        self.__find_events()
        self.event_iter = iter(self.events)

    def generate(self, mode='eval'):
        tracks = {}
        hits = {}

        try:
            event = next(self.event_iter)
        except:
            sys.stdout.write('\n')
            sys.stdout.flush()
            pinfo('Run out of particles')
            pinfo('Connecting to the next track database')
            self.__update_db()
            self.__make_event_iter()
            event = next(self.event_iter)

        runId, subRunId, eventId = event
        ptcls = self.session.query(Particle).filter(Particle.run==runId).\
            filter(Particle.subRun==subRunId).\
            filter(Particle.event==eventId).all()

        for ptcl in ptcls:
            strawHit_qrl = self.session.query(StrawHit).filter(StrawHit.particle==ptcl.id)
            hitNum = strawHit_qrl.count()

            if hitNum < self.hitNumCut:
                continue

            tracks[ptcl.id] = []
            track = tracks[ptcl.id]

            strawHits = strawHit_qrl.all()
            for hit in strawHits:
                track.append(hit.id)
                hits[hit.id] = (hit.x_reco, hit.y_reco, hit.z_reco, hit.t_reco)

            pdgId = self.session.query(Particle.pdgId).filter(Particle.id==ptcl.id).one_or_none()[0]
            track.append(pdgId)

        if mode == 'eval':
            return hits, tracks
        else:
            return hits


class Event_V2:
    def __init__(self, db_files, hitNumCut=0):
        self.dbs = db_files
        self.hitNumCut = hitNumCut

        self.db_iter = iter(db_files)

        self.current_db = None
        self.session = None
        self.__update_db()

        self.hitNums = []
        self.pdgIds = []
        self.strawHits = []

        self.hitNumIter = None
        self.pdgIdIter = None
        self.strawHitIter = None
        self.__make_iters()

    def __connect_db(self):
        engine = create_engine('sqlite:///'+str(self.current_db))
        Session = sessionmaker(bind=engine) # session factory
        session = Session() # session object
        self.session = session

    def __update_db(self):
        self.current_db = next(self.db_iter)
        self.__connect_db()

    def __make_iters(self):

        ptcls = self.session.query(Particle).order_by(Particle.run, Particle.subRun, Particle.event).all()

        event_tuples = []
        runs = self.session.query(Particle.run).distinct().order_by(Particle.run).all()
        pinfo("Getting all event identifications in the database")
        for run in runs:
            run = run[0]
            ptcl_subset = ptcls_qry.filter(Particle.run==run).all()
            subRuns = {ptcl.subRun for ptcl in ptcl_subset}
            for subRun in subRuns:
                ptcl_subset = self.session.query(Particle).filter(Particle.run==run).\
                    filter(Particle.subRun==subRun).all()
                events = {ptcl.event for ptcl in ptcl_subset}
                for event in events:
                    event_tuples.append((run, subRun, event))

        pinfo("Identifying Particles for each event")

        eventNum = len(event_tuples)
        idx = 0
        for runId, subRunId, eventId in event_tuples:
            sys.stdout.write(t_info(f'Parsing event: {idx+1}/{eventNum}', special='\r'))
            sys.stdout.flush()
            ptcls = self.session.query(Particle).filter(Particle.run==runId).\
                filter(Particle.subRun==subRunId).\
                filter(Particle.event==eventId).all()
            ptclIds = [ptcl.id for ptcl in ptcls]
            pdgIdsInWindow = [ptcl.pdgId for ptcl in ptcls]
            self.pdgIds.append(pdgIdsInWindow)

            hitNumsInWindow = []
            for ptclId in ptclIds:
                hitNum = self.session.query(StrawHit).filter(StrawHit.particle==ptclId).count()
                hitNumsInWindow.append(hitNum)
            self.hitNums.append(hitNumsInWindow)

            idx += 1

        sys.stdout.write('\n')
        sys.stdout.flush()

        pinfo("Constructing iterators")
        self.hitNumIter = iter(self.hitNums)
        self.pdgIdIter = iter(self.pdgIds)

        self.strawHits = self.session.query(StrawHit).order_by(StrawHit.ptcl).all()
        self.strawHitIter = iter(self.strawHits)
        return

    def generate(self, mode='eval'):
        tracks = {}
        hits = {}

        try:
            hitNumsInWindow = next(self.hitNumIter)
            pdgIdsInWindow = next(self.pdgIds)
        except:
            sys.stdout.write('\n')
            sys.stdout.flush()
            pinfo('Run out of particles')
            pinfo('Connecting to the next track database')
            self.__update_db()
            self.__find_iters()
            hitNumsInWindow = next(self.hitNumIter)
            pdgIdsInWindow = next(self.pdgIds)

        for trkIdx, hitNum in enumerate(hitNumsInWindow):
            if hitNum < self.hitNumCut:
                for i in range(hitNum):
                    next(strawHitIter)
                continue

            tracks[trkIdx] = []
            track = tracks[trkIdx]

            for i in range(hitNum):
                hit = next(strawHitIter)
                track.append(hit.id)
                hits[hit.id] = (hit.x_reco, hit.y_reco, hit.z_reco, hit.t_reco)

            pdgId = pdgIdsInWindow[trkIdx]
            track.append(pdgId)

        if mode == 'eval':
            return hits, tracks
        else:
            return hits
