# @Author: Billy Li <billyli>
# @Date:   11-03-2021
# @Email:  li000400@umn.edu
# @Last modified by:   billyli
# @Last modified time: 12-27-2021



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

        self.event_iter = None
        self.__make_ptclIdsIter()

    def __connect_db(self):
        engine = create_engine('sqlite:///'+str(self.current_db))
        Session = sessionmaker(bind=engine) # session factory
        session = Session() # session object
        self.session = session

    def __update_db(self):
        self.current_db = next(self.db_iter)
        self.__connect_db()

    def __find_ptclGroups(self):
        event_tuples = []
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
                    event_tuples.append((run, subRun, event))
        for runId, subRunId, eventId in event_tuples:
            ptcls = self.session.query(Particle).filter(Particle.run==runId).\
                filter(Particle.subRun==subRunId).\
                filter(Particle.event==eventId).all()

        ptcl_groups = self.session.query(Particle).group_by([Particle.run, Particle.subrun, Particle.event])
        ptclId_groups = [ [ptcl.id for ptcl in group] for group in ptcl_groups]

        self.ptclId_groups = []
        return

    def __make_ptclIdsIter(self):
        self.__find_ptclGroups()
        self.ptclIdsIter = iter(self.ptclId_groups)

    def generate(self, mode='eval'):
        tracks = {}
        hits = {}

        try:
            ptclIds = next(self.ptclIdsIter)
        except:
            sys.stdout.write('\n')
            sys.stdout.flush()
            pinfo('Run out of particles')
            pinfo('Connecting to the next track database')
            self.__update_db()
            self.__find_ptclGroups()
            ptclIds = next(self.ptclIdsIter)

        for ptclId in ptclIds:
            strawHit_qrl = self.session.query(StrawHit).filter(StrawHit.particle==ptclId)
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
