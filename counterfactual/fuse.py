import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.special import expit

def harmonic(a,b):
    zhm=expit(a)*expit(b)
#     return zhm/(1+zhm)
    return np.log(zhm/(1+zhm))

def sum_fuse(a,b):
    zsum=a+b
    zsum=expit(zsum)
#     return zsum
    return np.log(zsum)

def poe(a,b):
    return softmax(np.log(a)+np.log(b))
#     return np.log(zsum)

def add(a,b):
    return a + b

