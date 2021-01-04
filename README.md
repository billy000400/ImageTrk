# A Deep Learning Approach to Particle Reconstruction

# DLTracking
DLTracking is a deep learning package for particle reconstructions at the Mu2e
experiment.

## Status
The project is being irregularly updated due to the university winter vocation.


## Mu2e @ Minnesota

The muon-to-electron-conversion (Mu2e) experiment is going to search for the
neutrino-less muon-to-electron conversion. Such a conversion that violates
the charged-lepton flavor conservation is beyond the standard model, but is
predicted by many advanced theories, such as supersymmetry, Higgs doublets, and other
models explaining the neutrino mass hierarchy.

The Mu2e group at the University of Minnesota is working on Mu2e tracker's
assembly, quality assurance, and improvements. Besides, we are also discovering
 novel methods to conduct particle reconstruction.

This website/document is particularly for documenting the deep learning approaches
that we developed and evaluated for Mu2e's reconstruction missions.

---
## Prerequisite
### Download
Download the source code at [github](https://github.com/billy000400/DLTracking).

### FHiCL and C++
If you only use the sample track databases that we are studying, you could skip
 this section.

The C++ scripts in the src folder should be configured by the FHiCL files in the fcl folder.
A FHiCL file, as a configuration setup of C++ modules, should be run under
Fermilab's "art" event processing framework. These C++ scripts have special naming
rules and structures required by the art framework. You need to have basic
knowledge of the art framework and FHiCL before you write your C++ scripts and
FHiCL files.

You can know more about the art framework at
[here](https://art.fnal.gov/wp-content/uploads/2016/03/art-workbook-v0_91.pdf),
and know more about FHiCL at
[here](https://mu2ewiki.fnal.gov/wiki/FclIntro).



### Python
The software and packages used in this package and their versions are

- python==3.6
- SQLAlchemy==1.3.18
- numpy==1.18.1
- tensorflow_gpu==2.3.0
- art==4.7
- matplotlib==3.2.0
- pandas==1.1.0
- opencv_python==4.2.0.32
- Pillow==8.1.0
- scikit_learn==0.24.0
- tensorflow==2.4.0

---
## Acknowledgement
The package is developed by Haoyang(Billy) Li and advised by Professor
Ken Heller. Dr. Dan Ambrose and Kate Ciampa also gave me excellent inspirations
 and answered many questions about the simulated data.


## Support

If you have any questions or concerns, please contact Billy via li000400@umn.edu
