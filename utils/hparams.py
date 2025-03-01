import yaml

class hparam:
	def __init__(self, config: str = None, **kwargs):
		super().__init__()
		self.H = {}
		self.load_config(config)

	@property
	def keys(self) -> str:
		'''
		Print out hparams by calling "hparams.keys"
		'''
		keys = ""
		keys += "Hparams\n"
		for k, v in self.H.items():
			keys += f"{k}: {v}, " 
		return keys
			
	def load_config(self, config_path):
		try:
			with open(config_path, 'r', encoding='utf-8') as f:
				self.H.clear()
				self.H.update(yaml.safe_load(f))
				f.close()
			for k, v in self.H.items():
				self.__setattr__(k, v)
		except YAMLError as e:
			print(f"Unable to load config: {e}")

	def overwrite_key(self, key, new_value):
		try:
			self.H[key] = new_value
			self.__setattr__(key, new_value)
		except:
			print(f"Unable to update key: {key} to new value: {new_value}")