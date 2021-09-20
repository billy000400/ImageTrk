# @Author: Billy Li <billyli>
# @Date:   09-19-2021
# @Email:  li000400@umn.edu
# @Last modified by:   billyli
# @Last modified time: 09-19-2021



# This file has the mapped classes of the track database
# Which shows the structure of the database and the relationship between objects
# Author: Billy Haoyang Li
# Email: li000400@umn.edu

# Declare the base
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

# Construct classes
from sqlalchemy import Column, ForeignKey
from sqlalchemy.types import INTEGER, REAL

class Particle(Base):
    __tablename__ = "Particle"

    id = Column(INTEGER, primary_key=True, nullable=False)
    run = Column(INTEGER, nullable=False)
    subRun = Column(INTEGER, nullable=False)
    event = Column(INTEGER, nullable=False)
    track = Column(INTEGER, nullable=False)
    pdgId = Column(INTEGER, nullable=False)

class StrawDigiMC(Base):
    __tablename__ = "StrawDigiMC"

    id = Column(INTEGER, primary_key=True, nullable=False)
    particle = Column(INTEGER, ForeignKey("Particle.id"), nullable=False)
    x = Column(REAL, nullable=False)
    y = Column(REAL, nullable=False)
    z = Column(REAL, nullable=False)
    t = Column(REAL, nullable=False)
    p = Column(REAL, nullable=False)
    uniqueFace = Column(INTEGER, nullable=False)

class StrawHit(Base):
    __tablename__ = "StrawHit"

    id = Column(INTEGER, primary_key=True, nullable=False)
    particle = Column(INTEGER, ForeignKey("Particle.id"), nullable=False)
    strawDigiMC = Column(INTEGER, ForeignKey("StrawDigiMC.id"), nullable=False)
    x_reco = Column(REAL, nullable=False)
    y_reco = Column(REAL, nullable=False)
    z_reco = Column(REAL, nullable=False)
    t_reco = Column(REAL, nullable=False)
