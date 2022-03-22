from src.loader import DeepLoader
from src.trainer import Trainer
from src.classifier import DeepClassifier
import tensorflow as tf
import os
import numpy as np
import pandas as pd
tf.random.set_seed(13518136)
from src.util import construct_datetime, construct_time, save_model_with_pickle, dump_config
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from tensorflow.keras.callbacks import (
  ModelCheckpoint,
  LearningRateScheduler,
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

def scheduler(epoch, lr):
  return lr * tf.math.exp(-0.1)

class DeepTrainer(Trainer):
  def __init__(self, config_path):
    super().__init__(config_path)
    self.loader = DeepLoader(
      {**self.config["master"], **self.config["loader"]}
    )
    self.data = self.loader(self.config["master"]["sampling"])

  # Convert arr to label
  def get_label(self, arr):
    labels = self.config["master"]["labels"].split("_")
    idx = np.argmax(arr)
    return labels[idx]

  def convert_back_label(self, nparr):
    return [self.get_label(item) for item in nparr]

  def eval(self, y_true, y_pred):
    return {
      "pearson": pearsonr(y_true, y_pred)[0],
      "spearman": spearmanr(y_true, y_pred)[0],
      "mae": mean_absolute_error(y_true, y_pred),
      "mse": mean_squared_error(y_true, y_pred),
      "r2": r2_score(y_true, y_pred),
    }

  def fit(self):
    model = DeepClassifier(
      {**self.config["master"], **self.config["classifier"]}
    )

    # Compile model
    model.compile(
      loss=CategoricalCrossentropy(),
      optimizer=Adam(learning_rate=self.config["trainer"]["learning_rate"]),
    )

    # Define train callbacks
    callbacks = [
      LearningRateScheduler(scheduler),
      ModelCheckpoint(
        filepath=os.path.join(self.directory_path, "model-best.tf"),
        save_best_only=True,
        save_weights_only=True,
      ),
    ]

    # Fit Model
    model.fit(
      self.data["x_train"],
      self.data["y_train"],
      batch_size=self.config["trainer"]["batch_size"],
      epochs=self.config["trainer"]["epochs"],
      validation_data=(self.data["x_dev"], self.data["y_dev"]),
      callbacks=callbacks,
    )

    # Save latest weight
    model.save_weights(os.path.join(self.directory_path, "model-last.tf"))

    # print summary
    model.summary()

  def evaluate(self):
    model = DeepClassifier(
      {**self.config["master"], **self.config["classifier"]}
    )
    
    # Load best model
    model.load_weights(os.path.join(self.directory_path, "model-best.h5"))

    pred_train = self.model.predict(
      self.data["x_train"],
      batch_size=self.config["trainer"]["batch_size"],
      verbose=1,
    )
    pred_train = pd.DataFrame([pred_train], columns=["Prediction"])
    pred_train = pred_train.reset_index(drop=True)

    pred_dev = self.model.predict(
      self.data["x_dev"],
      batch_size=self.config["trainer"]["batch_size"],
      verbose=1,
    )
    pred_dev = pd.DataFrame([pred_dev], columns=["Prediction"])
    pred_dev = pred_dev.reset_index(drop=True)

    if not self.data["test"].empty:
      pred_test = self.model.predict(
        self.data["x_test"],
        batch_size=self.config["trainer"]["batch_size"],
        verbose=1,
      )
      pred_test = pd.DataFrame([pred_test], columns=["Prediction"])
      pred_test = pred_test.reset_index(drop=True)
    
    df_train = pd.concat([self.data["train"], pred_train], axis = 1)
    df_train["Prediction"] = df_train["Prediction"].apply(lambda x: self.get_label(x))
    
    df_dev = pd.concat([self.data["dev"], pred_dev], axis = 1)
    df_dev["Prediction"] = df_dev["Prediction"].apply(lambda x: self.get_label(x))
    
    df_test = pd.DataFrame([])
    if not self.data["test"].empty:
      df_test = pd.concat([self.data["test"], pred_test], axis = 1)


    return df_train, df_dev, df_test

  def save(self, df_train: pd.DataFrame, df_dev: pd.DataFrame, df_test: pd.DataFrame):
    # Save CSV
    df_train["Prediction"] = df_train["Prediction"].apply(lambda x: self.get_label(x))
    df_train.to_csv(os.path.join(self.directory_path, "pred_train.csv"), index=False)
    df_dev.to_csv(os.path.join(self.directory_path, "pred_dev.csv"), index=False)
    df_dev["Prediction"] = df_dev["Prediction"].apply(lambda x: self.get_label(x))
    if not df_test.empty:
      df_test["Prediction"] = df_test["Prediction"].apply(lambda x: self.get_label(x))
      df_test.to_csv(os.path.join(self.directory_path, "pred_test.csv"), index=False)


    # Log Result
    with open(os.path.join(self.directory_path, "log.txt"), "w+") as f:
      msg = "Experiment datetime: {}\n".format(datetime.now())
      msg += "Experiment prefix: {}\n".format(self.config["master"]["prefix"])
      msg += "Experiment description: {}\n".format(self.config["master"]["description"])
      msg += "Data Path: {}\n".format(self.config["loader"]["data_path"])
      msg += "Is sampling enabled: {}\n".format(self.config["master"]["sampling"])
      msg += "\n"
      msg += "========\t\t Tokenizer Details \t\t========\n\n"
      msg += "Tokenizer type: {}\n".format(self.config["loader"]["tokenizer_type"])
      msg += "Tokenizer model name: {}\n".format(self.config["loader"]["model_name"])
      msg += "Tokenizer max length: {}\n".format(self.config["loader"]["tokenizer_config"]["max_length"])
      msg += "\n"
      msg += "========\t\t Classifier Details \t\t========\n\n"
      msg += "Classifier type: {}\n".format(self.config["classifier"]["type"])
      msg += "Classifier model name: {}\n".format(self.config["classifier"]["svm_config"]["kernel"])
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
      msg += "========\t\t Classification Details Recap \t\t========\n\n"
      msg += "Overall predict time: {}\n".format(construct_time(self.classifier.overall_predict_time))
      msg += "Model score: {}\n".format(self.classifier.model_score)
      msg += "Labels:\n{}\n".format(self.config["master"]["labels"].split("_"))
      msg += "\nConfusion matrix:\n{}\n".format(self.classifier.confusion_matrix)
      msg += "\nClassification report:\n{}\n".format(self.classifier.classification_report)
      f.write(msg)
    dump_config(os.path.join(self.directory_path, "config.yaml"), self.config)
    print("Log and Model Successfully Saved to {}\n".format(self.directory_path))

def main(config):
  trainer = DeepTrainer(config)
  print("========\t\t Trainer is Fitting \t\t========")
  trainer.fit()
  print("========\t\t Trainer is Evaluating \t\t========")
  df_train, df_dev, df_test = trainer.evaluate()
  print("========\t\t Trainer is Wrapping Up \t\t========")
  trainer.save(df_train, df_dev, df_test)
  print("🚀🚀🚀🚀🚀🚀\t\t Trainer Flow Completed! \t\t🚀🚀🚀🚀🚀🚀")