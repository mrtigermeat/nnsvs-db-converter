from .pm import ParselmouthPE
from .pw import HarvestPE
from .rmvpe import RMVPE

def initialize_pe(hparams):
	if hparams.pe.lower() == 'parselmouth':
		return ParselmouthPE()
	elif hparams.pe.lower() == 'harvest':
		return HarvestPE()
	elif hparams.pe.lower() == 'rmvpe':
		return RMVPE(hparams.pe_ckpt)
	else:
		raise ValueError(f"Unknown PE: {hparams.pe}")