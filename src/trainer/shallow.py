from src.regressor import SVMRegressor
from src.embedder import WordVectorsEmbedder
from src.loader import ShallowLoader
from .trainer import Trainer
from src.util import construct_datetime, construct_time, save_model_with_pickle, dump_config
import os
from datetime import datetime

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
    if not self.config["embedder"]["is_pretrained"] or self.config["embedder"]["retrain"]:
      self.embedder.fit(self.data["merged"])
    print("Embedder Successfully Trained")

    # Train regressor
    train_key = self.config["master"]["train_key"]
    self.regressor.fit(self.data[train_key], self.embedder)
    print("\nRegressor Successfully Trained\n")

  def evaluate(self):
    test_key = self.config["master"]["test_key"]
    self.regressor.evaluate(self.data[test_key], self.embedder)
    print("Model Successfully Evaluated Data\n")

  def save(self):
    # Save Regressor (Model + Decomposer)
    save_model_with_pickle(self.regressor.model, os.path.join(self.directory_path, "regressor.model"))
    save_model_with_pickle(self.regressor.decomposer, os.path.join(self.directory_path, "decomposer.model"))
    # Save Embedder
    self.embedder.model.save(os.path.join(self.directory_path, self.config["embedder"]["model_type"]+".model"))
    # Save Prediction to CSV
    self.regressor.pred.to_csv(os.path.join(self.directory_path, "pred.csv"))
    # Log Result
    with open(os.path.join(self.directory_path, "log.txt"), "w+") as f:
      msg = "Experiment datetime: {}\n".format(datetime.now())
      msg += "Experiment Prefix: {}\n".format(self.config["master"]["prefix"])
      msg += "Experiment description: {}\n".format(self.config["master"]["description"])
      msg += "Data Path: {}\n".format(self.config["loader"]["data_path"])
      msg += "\n"
      msg += "========\t\t Embedder Details \t\t========\n\n"
      msg += "Embedding Path: {}\n".format(self.config["embedder"]["model_path"])
      msg += "Embedding Type: {}\n".format(self.config["embedder"]["model_type"])
      msg += "Embedding Behavior: {}\n".format(self.config["embedder"]["model_behavior"])
      msg += "Embedding vocab length: {}\n".format(len(self.embedder.model.wv))
      msg += "Embedding vector length: {}\n".format(self.embedder.model.wv.vector_size)
      msg += "Embedding window: {}\n".format(self.embedder.model.window)
      msg += "Embedding total train time: {}\n".format(construct_time(self.embedder.model.total_train_time))
      msg += "Embedding total train count: {}\n".format(self.embedder.model.train_count)
      msg += "Current train time: {}\n".format(construct_time(self.embedder.train_embedder_time))
      msg += "Total train sentences: {}\n".format(self.embedder.trained_with)
      msg += "\n"
      msg += "========\t\t Regressor Details \t\t========\n\n"
      msg += "Regressor type: {}\n".format(self.config["regressor"]["type"])
      msg += "Regressor kernel: {}\n".format(self.config["regressor"]["kernel"])
      msg += "Regressor gamma: {}\n".format(self.config["regressor"]["gamma"])
      msg += "Regressor max_iter: {}\n".format(self.config["regressor"]["max_iter"])
      msg += "Regressor degree: {}\n".format(self.config["regressor"]["degree"])
      msg += "Regressor train time: {}\n".format(construct_time(self.regressor.train_model_time))
      msg += "\n"
      msg += "========\t\t Decomposer Details \t\t========\n\n"
      msg += "Decomposer type: {}\n".format(self.config["regressor"]["decomposer"]["type"])
      msg += "Decomposer variance_tolerance: {}\n".format(self.config["regressor"]["decomposer"]["variance_tolerance"])
      msg += "Decomposer svd_solver: {}\n".format(self.config["regressor"]["decomposer"]["svd_solver"])
      msg += "Number of Features: {}\n".format(self.regressor.decomposer.n_features_)
      msg += "Number of Components: {}\n".format(self.regressor.decomposer.n_components_)
      msg += "Decomposer train time: {}\n".format(construct_time(self.regressor.train_decomposer_time))
      msg += "Total explained variance ratio: {}\n".format(sum(self.regressor.decomposer.explained_variance_ratio_))
      msg += "Noise Variance: {}\n".format(self.regressor.decomposer.noise_variance_)
      msg += "Decomposer total mean: {}\n".format(sum(self.regressor.decomposer.mean_))
      msg += "\n"
      msg += "========\t\t Classification Details Recap \t\t========\n\n"
      msg += "Overall Predict Time: {}\n".format(construct_time(self.regressor.overall_predict_time))
      msg += "Classifier Score: {}\n".format(self.regressor.model_score)
      msg += "Labels:\n{}\n".format(self.config["master"]["labels"].split("_"))
      msg += "\nConfusion matrix:\n{}\n".format(self.regressor.confusion_matrix)
      msg += "\nClassification Report:\n{}\n".format(self.regressor.classification_report)
      f.write(msg)
    dump_config(os.path.join(self.directory_path, "config.yaml"), self.config)
    print("Log and Model Successfully Saved to {}\n".format(self.directory_path))

def main(config):
  trainer = ShallowTrainer(config)
  print("========\t\t Trainer is Fitting \t\t========")
  trainer.fit()
  print("========\t\t Trainer is Evaluating \t\t========")
  trainer.evaluate()
  print("========\t\t Trainer is Wrapping Up \t\t========")
  trainer.save()
  print("🚀🚀🚀🚀🚀🚀\t\t Trainer Flow Completed! \t\t🚀🚀🚀🚀🚀🚀")