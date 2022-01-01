import pytest
import torch
import numpy as np
from scipy.signal import butter, lfilter, filtfilt
from hypercoil.nn.iirfilter import DTDF
from hypercoil.init.iirfilterbutforreal import IIRFilterSpec


class TestIIRFilter:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.Z = np.random.rand(10, 5, 100)
        self.Z2 = np.random.rand(10, 5, 100, 10)

        self.bw = butter(N=2, Wn=0.1, fs=1)
        self.bw1 = butter(N=1, Wn=0.1, fs=1)
        self.bw3 = butter(N=3, Wn=0.1, fs=1)
        self.bwbp = butter(N=2, Wn=(0.1, 0.2), fs=1, btype='bandpass')

    def compare_spec_and_ref(self, spec, ref):
        ff = DTDF(spec)
        out = ff(torch.tensor(self.Z)).detach()
        assert np.allclose(ref, out, atol=1e-6)

    def test_order2(self):
        ref = lfilter(self.bw[0], self.bw[1], self.Z)
        spec = IIRFilterSpec(N=2, Wn=0.1, fs=1, ftype='butter', btype='lowpass')
        self.compare_spec_and_ref(spec, ref)

    def test_order1(self):
        ref = lfilter(self.bw1[0], self.bw1[1], self.Z)
        spec = IIRFilterSpec(N=1, Wn=0.1, fs=1, ftype='butter', btype='lowpass')
        self.compare_spec_and_ref(spec, ref)

    def test_order3(self):
        ref = lfilter(self.bw3[0], self.bw3[1], self.Z)
        spec = IIRFilterSpec(N=3, Wn=0.1, fs=1, ftype='butter', btype='lowpass')
        self.compare_spec_and_ref(spec, ref)

    def test_bandpass(self):
        ref = lfilter(self.bwbp[0], self.bwbp[1], self.Z)
        spec = IIRFilterSpec(N=2, Wn=(0.1, 0.2), fs=1, ftype='butter', btype='bandpass')
        self.compare_spec_and_ref(spec, ref)
        #TODO: add test for filtering across another axis when it's implemented
        #ref2 = lfilter(bw[0], bw[1], Z2, axis=-2)
        #out2 = ff(torch.tensor(Z2), feature_ax=True)
