from src.util import get_config, get_latest_version
import os

class Trainer:
  def __init__(self, config_path):
    self.config = get_config(config_path)
    self.__init_folder()

  def __init_folder(self):
    save_path = self.config["master"]["save_path"]
    prefix = self.config["master"]["prefix"]
    release_mode = self.config["master"]["release_mode"]
    version = get_latest_version(save_path, prefix, mode=release_mode)
    self.directory_path = os.path.join(save_path, f"{prefix}-{version}")
    os.mkdir(self.directory_path)