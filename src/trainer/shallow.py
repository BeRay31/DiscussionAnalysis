from src.embedder import WordVectorsEmbedder
from src.loader import ShallowLoader
from .trainer import Trainer

class ShallowTrainer(Trainer):
  def __init__(self, config_path):
    super().__init__(config_path)
    self.loader = ShallowLoader(
      {**self.config["master"], **self.config["loader"]}
    )
    self.embedder = WordVectorsEmbedder(
      {**self.config["master"], **self.config["embedder"]}
    )
    self.regressor = None
  
  def fit(self):
    # load data
    self.data = self.loader()
    print(self.data["merged"].columns)

    # Train embedder
    if not self.config["embedder"]["is_pretrained"]:
      self.embedder.fit(self.data["merged"])
  
  def evaluate(self):
    pass

  def save(self):
    pass

def main(config):
  trainer = ShallowTrainer(config)
  trainer.fit()
  trainer.evaluate()
  trainer.save()