from .pm import ParselmouthPE
from .pw import HarvestPE

parselmouth_names = ['parselmouth', 'pm']
harvest_names = ['harvest', 'pw']

def initialize_pe(hparams):
	pe = hparams.pe
	pe_ckpt = hparams.pe_ckpt
	if pe.lower() in parselmouth_names:
		return ParselmouthPE()
	elif pe.lower() in harvest_names:
		return HarvestPE()
	else:
		raise ValueError(f"Unknown PE: {pe}")