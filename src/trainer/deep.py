from src.loader import DeepLoader
from src.trainer import Trainer

class DeepTrainer(Trainer):
  def __init__(self, config_path):
    super().__init__(config_path)
    self.loader = DeepLoader(
      {**self.config["master"], **self.config["loader"]}
    )
  
  def fit(self):
    pass

  def evaluate(self):
    pass

  def save(self):
    pass

def main(config):
  trainer = DeepTrainer(config)
  print("========\t\t Trainer is Fitting \t\t========")
  trainer.fit()
  print("========\t\t Trainer is Evaluating \t\t========")
  trainer.evaluate()
  print("========\t\t Trainer is Wrapping Up \t\t========")
  trainer.save()
  print("🚀🚀🚀🚀🚀🚀\t\t Trainer Flow Completed! \t\t🚀🚀🚀🚀🚀🚀")