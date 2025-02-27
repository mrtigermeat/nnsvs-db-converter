import pyworld as pw
import numpy as np

from .base_pe import PE
from utils import hparams

class HarvestPE(PE):
    def get_pitch(self, x, fs):
        time_step = hparams['time_step']
        f0_min = hparams['f0_min']
        f0_max = hparams['f0_max']
        voicing_threshold = hparams['voicing_threshold_midi']

        length = int(x.size / (time_step * fs))
        time_step *= 1000

        f0, _ = pw.harvest(x, fs, f0_floor=f0_min, f0_ceil=f0_max, frame_period=time_step)

        if f0.size < length:
            f0 = np.pad(f0, (0, length - f0.size))
        f0 = f0[:length]

        return f0