import yaml

hparams = {}

def set_hparams(config: str = "configs/base.yaml"):
	"""
	Load hparams. Referenced "hparams.py" from openvpi/DiffSinger
	"""
	hparams_ = load_config(config)

	global hparams

	hparams.clear()
	hparams.update(hparams_)

	return hparams_

def load_config(config_fn):
	try:
		with open(config_fn, 'r', encoding='utf-8') as f:
			hparams_ = yaml.safe_load(f)
	except:
		print(f"Unable to laod config: {config_fn}")

	return hparams_

set_hparams()