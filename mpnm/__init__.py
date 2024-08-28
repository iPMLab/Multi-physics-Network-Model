'''
This package contains the classes and functions necessary to run the mpnm algorithm.
'''


import numpy as np
from mpnm._algorithm import algorithm
from mpnm._Base import Base
from mpnm._network import network
from mpnm._topotools import topotools
from mpnm._phase import phase
np.seterr(divide='ignore', invalid='ignore')
# __all__=['_algorithm','_Base','_network','_phase','_topotools']