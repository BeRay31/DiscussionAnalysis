from src.classifier import ShallowClassifier
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
    self.classifier = ShallowClassifier(
      {**self.config["master"], **self.config["classifier"]}
    )
    self.data = self.loader()
  
  def fit(self):
    # Train embedder
    if not self.config["embedder"]["is_pretrained"] or self.config["embedder"]["retrain"]:
      self.embedder.fit(self.data["merged"])
    print("Embedder Successfully Trained")

    # Train classifier
    train_key = self.config["master"]["train_key"]
    self.classifier.fit(self.data[train_key], self.embedder)
    print("\nClassifier Successfully Trained\n")

  def evaluate(self):
    train_key = self.config["master"]["train_key"]
    dev_key = self.config["master"]["dev_key"]
    test_key = self.config["master"]["test_key"]
    
    train_pred = self.classifier.evaluate(self.data[train_key], self.embedder)
    dev_pred = self.classifier.evaluate(self.data[dev_key], self.embedder)
    if not self.data[test_key].empty:
      test_pred = self.classifier.evaluate(self.data[test_key], self.embedder)
    test_pred = {}
    print("Model Successfully Evaluated Data\n")

    return {
      f"{train_key}": train_pred,
      f"{dev_key}": dev_pred,
      f"{test_key}": test_pred
    }

  def save(self, eval_result):
    train_key = self.config["master"]["train_key"]
    dev_key = self.config["master"]["dev_key"]
    test_key = self.config["master"]["test_key"]

    # Save Classifier (Model + Decomposer)
    save_model_with_pickle(self.classifier.model, os.path.join(self.directory_path, "classifier.model"))
    save_model_with_pickle(self.classifier.decomposer, os.path.join(self.directory_path, "decomposer.model"))

    # Save Embedder
    self.embedder.model.save(os.path.join(self.directory_path, self.config["embedder"]["model_type"]+".model"))
    
    train_pred = eval_result[f"{train_key}"]
    dev_pred = eval_result[f"{dev_key}"]
    test_pred = eval_result[f"{test_key}"]

    # Save Prediction to CSV
    train_pred["prediction"].to_csv(os.path.join(self.directory_path, "train_pred.csv"), index=False)
    dev_pred["prediction"].to_csv(os.path.join(self.directory_path, "dev_pred.csv"), index=False)
    if len(test_pred.keys()) > 0:
      test_pred["prediction"].to_csv(os.path.join(self.directory_path, "test_pred.csv"), index=False)
    
    # Log Result
    with open(os.path.join(self.directory_path, "log.txt"), "w+") as f:
      msg = "Experiment datetime: {}\n".format(datetime.now())
      msg += "Experiment prefix: {}\n".format(self.config["master"]["prefix"])
      msg += "Experiment description: {}\n".format(self.config["master"]["description"])
      msg += "Data Path: {}\n".format(self.config["loader"]["data_path"])
      msg += "\n"
      msg += "========\t\t Embedder Details \t\t========\n\n"
      msg += "Embedding path: {}\n".format(self.config["embedder"]["model_path"])
      msg += "Embedding type: {}\n".format(self.config["embedder"]["model_type"])
      msg += "Embedding behavior: {}\n".format(self.config["embedder"]["model_behavior"])
      msg += "Embedding vocab length: {}\n".format(len(self.embedder.model.wv))
      msg += "Embedding vector length: {}\n".format(self.embedder.model.wv.vector_size)
      msg += "Embedding window: {}\n".format(self.embedder.model.window)
      msg += "Embedding total train time: {}\n".format(construct_time(self.embedder.model.total_train_time))
      msg += "Embedding total train count: {}\n".format(self.embedder.model.train_count)
      msg += "Current train time: {}\n".format(construct_time(self.embedder.train_embedder_time))
      msg += "Total train sentences: {}\n".format(self.embedder.trained_with)
      msg += "\n"
      msg += "========\t\t Classifier Details \t\t========\n\n"
      msg += "Classifier type: {}\n".format(self.config["classifier"]["type"])
      msg += "Classifier kernel: {}\n".format(self.config["classifier"]["svm_config"]["kernel"])
      msg += "Classifier gamma: {}\n".format(self.config["classifier"]["svm_config"]["gamma"])
      msg += "Classifier max_iter: {}\n".format(self.config["classifier"]["svm_config"]["max_iter"])
      msg += "Classifier degree: {}\n".format(self.config["classifier"]["svm_config"]["degree"])
      msg += "Classifier train time: {}\n".format(construct_time(self.classifier.train_model_time))
      msg += "\n"
      msg += "========\t\t Decomposer Details \t\t========\n\n"
      msg += "Decomposer type: {}\n".format(self.config["classifier"]["decomposer"]["type"])
      msg += "Decomposer variance_tolerance: {}\n".format(self.config["classifier"]["decomposer"]["decomposer_config"]["n_components"])
      msg += "Decomposer svd_solver: {}\n".format(self.config["classifier"]["decomposer"]["decomposer_config"]["svd_solver"])
      msg += "Number of features: {}\n".format(self.classifier.decomposer.n_features_)
      msg += "Number of components: {}\n".format(self.classifier.decomposer.n_components_)
      msg += "Decomposer train time: {}\n".format(construct_time(self.classifier.train_decomposer_time))
      msg += "Total explained variance ratio: {}\n".format(sum(self.classifier.decomposer.explained_variance_ratio_))
      msg += "Noise variance: {}\n".format(self.classifier.decomposer.noise_variance_)
      msg += "Decomposer total mean: {}\n".format(sum(self.classifier.decomposer.mean_))
      msg += "\n"
      msg += "========\t\t Train Data Classification Details Recap \t\t========\n\n"
      msg += "Predict time: {}\n".format(construct_time(train_pred["predict_time"]))
      msg += "Score: {}\n".format(train_pred["score"])
      msg += "Labels:\n{}\n".format(self.config["master"]["labels"].split("_"))
      msg += "\nConfusion matrix:\n{}\n".format(train_pred["confusion_matrix"])
      msg += "\nClassification report:\n{}\n".format(train_pred["classification_report"])
      msg += "\n"
      msg += "========\t\t Dev Data Classification Details Recap \t\t========\n\n"
      msg += "Predict time: {}\n".format(construct_time(dev_pred["predict_time"]))
      msg += "Score: {}\n".format(dev_pred["score"])
      msg += "Labels:\n{}\n".format(self.config["master"]["labels"].split("_"))
      msg += "\nConfusion matrix:\n{}\n".format(dev_pred["confusion_matrix"])
      msg += "\nClassification report:\n{}\n".format(dev_pred["classification_report"])
      msg += "\n"
      if len(test_pred.keys()) > 0:
        msg += "========\t\t Test Data Classification Details Recap \t\t========\n\n"
        msg += "Predict time: {}\n".format(construct_time(test_pred["predict_time"]))
        msg += "Score: {}\n".format(test_pred["score"])
        msg += "Labels:\n{}\n".format(self.config["master"]["labels"].split("_"))
        msg += "\nConfusion matrix:\n{}\n".format(test_pred["confusion_matrix"])
        msg += "\nClassification report:\n{}\n".format(test_pred["classification_report"])
        msg += "\n"
      f.write(msg)
    dump_config(os.path.join(self.directory_path, "config.yaml"), self.config)
    print("Log and Model Successfully Saved to {}\n".format(self.directory_path))

def main(config):
  trainer = ShallowTrainer(config)
  print("========\t\t Trainer is Fitting \t\t========")
  trainer.fit()
  print("========\t\t Trainer is Evaluating \t\t========")
  res = trainer.evaluate()
  print("========\t\t Trainer is Wrapping Up \t\t========")
  trainer.save(res)
  print("🚀🚀🚀🚀🚀🚀\t\t Trainer Flow Completed! \t\t🚀🚀🚀🚀🚀🚀")