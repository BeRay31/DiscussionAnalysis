from src.util import get_config, get_latest_version
from constant import Constant
import os
class Trainer:
  def __init__(self, config_path, prefix):
    self.config = get_config(config_path)
    self.prefix = prefix
    self.loader = None
    self.__init_folder()

  def __init_folder(self):
    version = get_latest_version(Constant.MODELS, self.prefix, mode=self.config["release-mode"])
    self.directory_path = os.path.join(Constant.MODELS, f"{self.prefix}-{version}")
    os.mkdir(self.directory_path)
  
  def fit(self):
    # Load Data
    self.data = self.loader()