# @Author: Billy Li <billyli>
# @Date:   11-03-2021
# @Email:  li000400@umn.edu
# @Last modified by:   billyli
# @Last modified time: 01-11-2022



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
    def __init__(self, db_files, hitNumCut=0, eventNum=None):
        self.dbs = db_files
        self.hitNumCut = hitNumCut

        self.db_iter = iter(db_files)

        self.current_db = None
        self.session = None
        self.__update_db()

        self.eventNum = eventNum
        self.__count_event()

        self.ptclNums = []
        self.pdgIds = []
        self.strawHits = []

        self.ptclNumIter = None
        self.pdgIdIter = None
        # self.strawHitIter = None
        self.__make_iters()

        self.current_ptclId = 1
        self.current_hitId = 1

    def __connect_db(self):
        engine = create_engine('sqlite:///'+str(self.current_db))
        Session = sessionmaker(bind=engine) # session factory
        session = Session() # session object
        self.session = session

    def __update_db(self):
        self.current_db = next(self.db_iter)
        self.__connect_db()

    def __count_event(self):
        eventNumMax = self.session.query(Particle.run, Particle.subRun, Particle.event).distinct().count()
        print(eventNumMax)
        if self.eventNum is None:
            self.eventNum = eventNumMax
        elif eventNumMax < self.eventNum:
            pwarn(f"The required event number: {self.eventNum}\
             is greater than the maximum: {eventNumMax}")
            pwarn(f"Changing event number to maximum")
            self.eventNum = eventNumMax

    def __make_iters(self):

        ptcls = self.session.query(Particle).order_by(Particle.run, Particle.subRun, Particle.event).all()
        ptclIter = iter(ptcls)

        runs = self.session.query(Particle.run).distinct().order_by(Particle.run).all()
        scanEvtNum = 0
        for run in runs:
            if ((self.eventNum!=None) and (scanEvtNum>=self.eventNum)):
                break
            run = run[0]
            subRuns = self.session.query(Particle.subRun)\
                .filter(Particle.run==run)\
                .distinct()\
                .order_by(Particle.subRun)\
                .all()
            for subRun in subRuns:
                if ((self.eventNum!=None) and (scanEvtNum>=self.eventNum)):
                    break
                subRun = subRun[0]
                events = self.session.query(Particle.event)\
                    .filter(Particle.run==run)\
                    .filter(Particle.subRun==subRun)\
                    .distinct()\
                    .order_by(Particle.event)\
                    .all()
                for event in events:
                    if ((self.eventNum!=None) and (scanEvtNum>=self.eventNum)):
                        break
                    sys.stdout.write(t_info(f'Getting ids for event: {scanEvtNum+1}/{self.eventNum}', special='\r'))
                    sys.stdout.flush()
                    event = event[0]
                    ptclsInWindow = self.session.query(Particle)\
                        .filter(Particle.run==run)\
                        .filter(Particle.subRun==subRun)\
                        .filter(Particle.event==event).all()
                    ptclNumInWindow = len(ptclsInWindow)
                    self.ptclNums.append(ptclNumInWindow)
                    self.pdgIds.append([ptcl.pdgId for ptcl in ptclsInWindow])
                    scanEvtNum += 1
        sys.stdout.write('\n')
        sys.stdout.flush()


        pinfo("Constructing iterators")
        self.ptclNumIter = iter(self.ptclNums)
        self.pdgIdIter = iter(self.pdgIds)

        self.strawHits = self.session.query(StrawHit).order_by(StrawHit.particle).all()
        # self.strawHitIter = iter(self.strawHits)
        return

    def generate(self, mode='eval'):
        tracks = {}
        hits = {}

        try:
            ptclNumInWindow = next(self.ptclNumIter)
            pdgIdsInWindow = next(self.pdgIdIter)
        except:
            sys.stdout.write('\n')
            sys.stdout.flush()
            pinfo('Run out of particles')
            pinfo('Connecting to the next track database')
            self.__update_db()
            self.__find_iters()
            ptclNumInWindow = next(self.ptclNumIter)
            pdgIdsInWindow = next(self.pdgIdIter)

        pdgIdIter = iter(pdgIdsInWindow)

        trkIdx = 0
        for i in range(ptclNumInWindow):
            hitsPerPtcl = []
            while self.strawHits[self.current_hitId].particle == self.current_ptclId:
                hit = self.strawHits[self.current_hitId]
                hitsPerPtcl.append(hit)
                self.current_hitId += 1
            self.current_ptclId += 1
            hitNum = len(hitsPerPtcl)
            if hitNum < self.hitNumCut:
                next(pdgIdIter)
                continue
            tracks[trkIdx] = []
            track = tracks[trkIdx]
            for hit in hitsPerPtcl:
                track.append(hit.id)
                hits[hit.id] = (hit.x_reco, hit.y_reco, hit.z_reco, hit.t_reco)

            pdgId = next(pdgIdIter)
            track.append(pdgId)
            trkIdx += 1

        if mode == 'eval':
            return hits, tracks
        else:
            return hits
