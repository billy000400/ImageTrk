# A Deep Learning Approach to Particle Reconstruction

# DLTracking
DLTracking is a deep learning package for particle reconstructions at the Mu2e
experiment.

## Approach
Given a cluster of digitized detector signals, we project their reconstructed positions onto the XY plane.

![A plot of raw data](/home/Billy/Mu2e/analysis/DLTracking/pictures/FRCNN_raw.png)

Next, we apply the Faster R-CNN to localize and draw a bounding box around every track. Every bounding box contains a smaller cluster of signals, which includes both signals of the main track in the bounding box and the signals of other bounding boxes' main tracks.

![FRCNN](/home/Billy/Mu2e/analysis/DLTracking/pictures/FRCNN.png)

After the Faster R-CNN, we resize every bounding boxes to a 256x256 image and utilize a semantic image segmentation model to extract the main track's signals from the smaller cluster. As shown below, our image segmentation model assigns each pixel in a grayscale image to a RGB vector. The grayscale image on the left side represents the signal's number density in pixels, and the RGB image on the right side represents the predicted categories of pixels.Red pixels belong to the bounding box's major track, while blue cells belong to other bounding boxes' major tracks.

![Extractor](/home/Billy/Mu2e/analysis/DLTracking/pictures/Extractor.png)

In short, we call red pixels "major" and call blue pixels "background." The white pixels are called "blank" because they do not contain any signals.

Now, the digitized detector signals have been grouped into tracks. Further analysis of momentum and energy can be conducted.

## Status
We are actively researching and developing this package. The updates will first
be shown in our GitHub repository and then documented on this website.   

Currently, we are looking for the optimal scales of hyperparameters to regularize our
track extractor. Current data shows that different models, such as
ResNet, U-Net, and their hybrids, only have a difference of 1-3% to the extractor's performance,

|        Model        | ResNet | U-Net | U-ResNet |
|:-------------------:|:------:|:-----:|:--------:|
| Validation accuracy |  81.5  |  83.6 |   84.1   |

while the 'correct' depth and regularizations can improve
the performance by about 39%.   

|     Dropout rate    |  0 | 0.26 | 0.64 | 0.85 |
|:-------------------:|:--:|:----:|:----:|:----:|
| Validation accuracy | 50 |  87  |  89  | 87.5 |

The interesting part of this process is that applying dropout layers here is not just to prevent overfitting. As the "Dropout rate and Validation accuracy" table shows, an intermediate dropout rate let the model generalize better on the validation data even if the model is still significantly overfitted.

We can use an analogy to explain this. Suppose we have a linear signal plus a small sinusoidal signal. If we train a 100th-degree polynomial with very limited data and no regularization, our model will generalize poor on new data. However, if we train a 5th degree polynomial with limited data and very strong regularizations on higher order terms, the model will treat the sinusoidal signal as random noise and prefer to draw a straight line through the data. The second model will have a better generalization on new data, but it's not what we want. We want to have a model with enough parameters to be an approximator of some higher-order functions and does not rote memorize the training set to improve the training accuracy.

Thus, the goal of finding the optimal dropout rate is to find the best balance between not treating variant signals as noise and not remembering the 'textbook' to improve scores. Below is the learning curve of a ResNet decoder-encoder that has about 20,000,000 parameters, trained with a 0.64 dropout rate.

![](/home/Billy/Mu2e/analysis/DLTracking/pictures/0.64.png)

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
