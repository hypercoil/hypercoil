import torch
import numpy as np
from scipy.signal import butter, lfilter, filtfilt
from hypercoil.nn.iirfilter import DTDF
from hypercoil.init.iirfilterbutforreal import IIRFilterSpec


Z = np.random.rand(10, 5, 100)
Z2 = np.random.rand(10, 5, 100, 10)


bw = butter(N=2, Wn=0.1, fs=1)
bw1 = butter(N=1, Wn=0.1, fs=1)
bw3 = butter(N=3, Wn=0.1, fs=1)
ref = lfilter(bw[0], bw[1], Z)
ref1 = lfilter(bw1[0], bw1[1], Z)
ref2 = lfilter(bw[0], bw[1], Z2, axis=-2)
ref3 = lfilter(bw3[0], bw3[1], Z)


bwX = IIRFilterSpec(N=2, Wn=0.1, fs=1, ftype='butter', btype='lowpass')
ff = DTDF(bwX)
out = ff(torch.tensor(Z))

bw1X = IIRFilterSpec(N=1, Wn=0.1, fs=1, ftype='butter', btype='lowpass')
ff1 = DTDF(bw1X)
out1 = ff1(torch.tensor(Z))

bw3X = IIRFilterSpec(N=3, Wn=0.1, fs=1, ftype='butter', btype='lowpass')
ff3 = DTDF(bw3X)
out3 = ff3(torch.tensor(Z))

out2 = ff(torch.tensor(Z2), feature_ax=True)
