from src.loader import DeepLoader
from src.trainer import Trainer

class DeepTrainer(Trainer):
  def __init__(self, config_path):
    super().__init__(config_path)
    self.loader = DeepLoader(
      {**self.config["master"], **self.config["loader"]}
    )