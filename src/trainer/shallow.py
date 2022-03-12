from src.regressor import SVMRegressor
from src.embedder import WordVectorsEmbedder
from src.loader import ShallowLoader
from .trainer import Trainer
import pickle

class ShallowTrainer(Trainer):
  def __init__(self, config_path):
    super().__init__(config_path)
    self.loader = ShallowLoader(
      {**self.config["master"], **self.config["loader"]}
    )
    self.embedder = WordVectorsEmbedder(
      {**self.config["master"], **self.config["embedder"]}
    )
    self.regressor = SVMRegressor(
      {**self.config["master"], **self.config["regressor"]}
    )
    self.data = self.loader()
  
  def fit(self):
    # Train embedder
    if not self.config["embedder"]["is_pretrained"]:
      self.embedder.fit(self.data["merged"])

    # Train regressor
    train_key = self.config["train_key"]
    self.regressor.fit(self.data[train_key], self.embedder)

  def evaluate(self):
    test_key = self.config["test_key"]
    self.regressor.evaluate(self.data[test_key], self.embedder)

  def save(self):
    pass

def main(config):
  trainer = ShallowTrainer(config)
  print("========\t\tTrainer is Fitting\t\t========")
  trainer.fit()
  print("========\t\tTrainer is Evaluating\t\t========")
  trainer.evaluate()
  print("========\t\tTrainer is Wrapping Up\t\t========")
  trainer.save()