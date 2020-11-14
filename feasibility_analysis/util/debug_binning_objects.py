from abstract import binning_objects
from mu2e_output import*
import numpy as np

step = 1620/800
xbins = [ -810+i*step for i in range(801) ]
object = [(336.91152955737743, 290.41331146894385, -100.03720548581987)]
feature = [336.91152955737743]
result = binning_objects(object, feature, xbins)
pdebug(result)
