import parselmouth as pm
import numpy as np

from .base_pe import PE

class ParselmouthPE(PE):
    def get_pitch(self, x, fs, hparams):
        # from openvpi/DiffSinger/utils/binarizer_utils.py get_pitch_parselmouth
        time_step = hparams.time_step
        f0_min = hparams.f0_min
        f0_max = hparams.f0_max
        voicing_threshold = hparams.voicing_threshold_midi

        hop_size = time_step * fs
        length = int(x.size / hop_size)

        l_pad = int(np.ceil(1.5 / f0_min * fs))
        r_pad = int(hop_size * ((x.size - 1) // hop_size + 1) - x.size + l_pad + 1)
        x = np.pad(x, (l_pad, r_pad))

        p = pm.Sound(x, sampling_frequency=fs).to_pitch_ac(
            time_step=time_step, voicing_threshold=voicing_threshold,
            pitch_floor=f0_min, pitch_ceiling=f0_max)
        assert np.abs(p.t1 - 1.5 / f0_min) < 0.001

        f0 = p.selected_array['frequency']
        if f0.size < length:
            f0 = np.pad(f0, (0, length - f0.size))
        f0 = f0[:length]

        return f0