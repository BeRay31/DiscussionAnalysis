from this import d
from src.loader import DeepLoader
from src.trainer import Trainer
from src.classifier import DeepClassifier
from src.util import get_latest_version
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from time import time

tf.random.set_seed(13518136)
from src.util import construct_time, dump_config, write_dict
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from keras.callbacks import (
  ModelCheckpoint,
  LearningRateScheduler,
)
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report


def scheduler(epoch, lr):
  return lr * tf.math.exp(-0.1)

class DeepTrainer(Trainer):
  def __init__(self, config_path):
    super().__init__(config_path)
    self.loader = DeepLoader(
      {**self.config["master"], **self.config["loader"]}
    )
    self.data = self.loader(self.config["master"]["sampling"])
    self.models_path = None
    self.eval_path = None
    self.overall_time_train = None

  # Convert arr to label
  def get_label(self, arr):
    labels = self.config["master"]["labels"].split("_")
    idx = np.argmax(arr)
    return labels[idx]

  def evalMetrics(self, y_true, y_pred):
    return {
      "pearson": pearsonr(y_true, y_pred)[0],
      "spearman": spearmanr(y_true, y_pred)[0],
      "mae": mean_absolute_error(y_true, y_pred),
      "mse": mean_squared_error(y_true, y_pred),
      "r2": r2_score(y_true, y_pred),
    }

  def fit(self):
    # mkdir for models
    self.models_path = os.path.join(self.directory_path, "models")
    os.mkdir(self.models_path)

    self.model = DeepClassifier(
      {**self.config["master"], **self.config["classifier"]}
    )

    # Compile model
    self.model.compile(
      loss=CategoricalCrossentropy(),
      optimizer=Adam(learning_rate=self.config["trainer"]["learning_rate"]),
    )

    # Define train callbacks
    callbacks = [
      LearningRateScheduler(scheduler),
      ModelCheckpoint(
        filepath=os.path.join(self.models_path, 
        f"{get_latest_version(self.models_path, 'model_epoch-', suffix='dlw')}"),
        save_weights_only=True
      )
    ]

    start = time()
    # Fit Model
    self.model.fit(
      self.data["x_train"],
      self.data["y_train"],
      batch_size=self.config["trainer"]["batch_size"],
      epochs=self.config["trainer"]["epochs"],
      validation_data=(self.data["x_dev"], self.data["y_dev"]),
      callbacks=callbacks,
    )
    end = time()
    self.overall_time_train = round(end - start, 2)

    # print summary
    self.model.summary()

  def evaluate(self):
    # prepare dir for evaluation
    self.eval_path = os.path.join(self.directory_path, "evaluation")
    os.mkdir(self.eval_path)

    model_weights = [weight for weight in os.listdir(self.models_path)]
    for model_weight in model_weights:
      # Load best model
      self.model.load_weights(os.path.join(self.models_path, model_weight))
      
      start = time()
      pred_train = self.model.predict(
        self.data["x_train"],
        batch_size=self.config["trainer"]["batch_size"],
        verbose=1,
      )
      end = time()
      pred_train = pd.DataFrame([pred_train], columns=["Prediction"])
      pred_train = pred_train.reset_index(drop=True)
      train_pred_time = round(end - start, 2)

      start = end
      pred_dev = self.model.predict(
        self.data["x_dev"],
        batch_size=self.config["trainer"]["batch_size"],
        verbose=1,
      )
      end = time()
      pred_dev = pd.DataFrame([pred_dev], columns=["Prediction"])
      pred_dev = pred_dev.reset_index(drop=True)
      dev_pred_time = round(end - start, 2)

      test_pred_time = None
      if not self.data["test"].empty:
        start = end
        pred_test = self.model.predict(
          self.data["x_test"],
          batch_size=self.config["trainer"]["batch_size"],
          verbose=1
        )
        end = time()
        pred_test = pd.DataFrame([pred_test], columns=["Prediction"])
        pred_test = pred_test.reset_index(drop=True)
        test_pred_time = round(end - start, 2)

      df_train = pd.concat([self.data["train"], pred_train], axis = 1)
      df_dev = pd.concat([self.data["dev"], pred_dev], axis = 1)
      df_test = pd.DataFrame([])

      if not self.data["test"].empty:
        df_test = pd.concat([self.data["test"], pred_test], axis = 1)

      self.save(df_train, df_dev, df_test, train_pred_time, dev_pred_time, test_pred_time, model_weight)

  def save(self, df_train: pd.DataFrame, df_dev: pd.DataFrame, df_test: pd.DataFrame, train_pred_time, dev_pred_time, test_pred_time, model_name):
    # Prepare Folder
    model_names = model_name.split(".")
    model_name = ".".join(model_names[:len(model_names) - 1])
    save_path = os.path.join(self.eval_path, model_name)
    os.makedirs(save_path)

    labels = self.config["master"]["labels"].split("_")
    # Eval Metrics
    train_eval_metrics = self.evalMetrics(
      y_pred=df_train["Prediction"].values,
      y_true=self.data["y_train"]
    )
    
    dev_eval_metrics = self.evalMetrics(
      y_pred=df_dev["Prediction"].values,
      y_true=self.data["y_dev"]
    )

    test_eval_metrics = {}
    if not df_test.empty:
      test_eval_metrics = self.evalMetrics(
        y_pred=df_test["Prediction"].values,
        y_true=self.data["y_test"]
      )
    
    # Save CSV
    df_train["Prediction"] = df_train["Prediction"].apply(lambda x: self.get_label(x))
    train_confusion_matrix = confusion_matrix(
      y_true=df_train[self.config["master"]["label_key"]],
      y_pred=df_train["Prediction"],
      labels=labels
    )
    train_classification_report = classification_report(
      y_true=df_train[self.config["master"]["label_key"]],
      y_pred=df_train["Prediction"],
      labels=labels
    )

    df_dev["Prediction"] = df_dev["Prediction"].apply(lambda x: self.get_label(x))
    dev_confusion_matrix = confusion_matrix(
      y_true=df_dev[self.config["master"]["label_key"]],
      y_pred=df_dev["Prediction"],
      labels=labels
    )
    dev_classification_report = classification_report(
      y_true=df_dev[self.config["master"]["label_key"]],
      y_pred=df_dev["Prediction"],
      labels=labels
    )

    df_train.to_csv(os.path.join(save_path, "pred_train.csv"), index=False)
    df_dev.to_csv(os.path.join(save_path, "pred_dev.csv"), index=False)
    if not df_test.empty:
      df_test["Prediction"] = df_test["Prediction"].apply(lambda x: self.get_label(x))
      df_test.to_csv(os.path.join(save_path, "pred_test.csv"), index=False)
      test_confusion_matrix = confusion_matrix(
        y_true=df_test[self.config["master"]["label_key"]],
        y_pred=df_test["Prediction"],
        labels=labels
      )
      test_classification_report = classification_report(
        y_true=df_test[self.config["master"]["label_key"]],
        y_pred=df_test["Prediction"],
        labels=labels
      )

    # Log Result
    with open(os.path.join(save_path, "log.txt"), "w+") as f:
      msg = "Experiment datetime: {}\n".format(datetime.now())
      msg += "Experiment prefix: {}\n".format(self.config["master"]["prefix"])
      msg += "Experiment description: {}\n".format(self.config["master"]["description"])
      msg += "Data Path: {}\n".format(self.config["loader"]["data_path"])
      msg += "Is sampling enabled: {}\n".format(self.config["master"]["sampling"])
      msg += "Training batch size: {}\n".format(self.config["trainer"]["batch_size"])
      msg += "Training learning rate: {}\n".format(self.config["trainer"]["learning_rate"])
      msg += "Training Epochs: {}\n".format(self.config["trainer"]["epochs"])
      msg += "\n"
      msg += "========\t\t Tokenizer Details \t\t========\n\n"
      msg += "Tokenizer type: {}\n".format(self.config["loader"]["tokenizer_type"])
      msg += "Tokenizer model name: {}\n".format(self.config["loader"]["model_name"])
      msg += "Tokenizer max length: {}\n".format(self.config["loader"]["tokenizer_config"]["max_length"])
      msg += "\n"
      msg += "========\t\t Classifier Details \t\t========\n\n"
      msg += "Classifier type: {}\n".format(self.config["classifier"]["type"])
      msg += "Classifier model name: {}\n".format(self.config["classifier"]["model_name"])
      msg += "\n"
      msg += "========\t\t Recurrent Layer Details \t\t========\n\n"
      msg += "Recurrent layer type: {}\n".format(self.config["classifier"]["recurrent_layer"])
      msg += "Recurrent layer number of unit: {}\n".format(self.config["classifier"]["recurrent_unit"])
      msg += "Recurrent layer dropout rate: {}\n".format(self.config["classifier"]["recurrent_dropout"])
      msg += "\n"
      msg += "========\t\t Classification Details Recap \t\t========\n\n"
      msg += "Model name: {}\n".format(model_name)
      msg += "Overall train time: {}\n".format(construct_time(self.overall_time_train))
      msg += "Predict train data time: {}\n".format(construct_time(train_pred_time))
      msg += "Predict dev data time: {}\n".format(construct_time(dev_pred_time))
      msg += "Predict test data time: {}\n".format(construct_time(test_pred_time))
      msg += "Labels:\n{}\n".format(self.config["master"]["labels"].split("_"))
      msg += "\n"
      msg += "========\t\t Prediction Metrics Details Recap \t\t========\n\n"
      msg += "Train\n{}\n\n".format(write_dict(train_eval_metrics))
      msg += "\nConfusion matrix:\n{}\n".format(train_confusion_matrix)
      msg += "\nClassification report:\n{}\n".format(train_classification_report)
      msg += "Dev\n{}\n\n".format(write_dict(dev_eval_metrics))
      msg += "\nConfusion matrix:\n{}\n".format(dev_confusion_matrix)
      msg += "\nClassification report:\n{}\n".format(dev_classification_report)
      msg += "Test\n{}\n\n".format(write_dict(test_eval_metrics))
      msg += "\nConfusion matrix:\n{}\n".format(test_confusion_matrix)
      msg += "\nClassification report:\n{}\n".format(test_classification_report)
      f.write(msg)
    dump_config(os.path.join(save_path, "config.yaml"), self.config)
    print("Log and Model Successfully Saved to {}\n".format(save_path))

def main(config):
  trainer = DeepTrainer(config)
  print("========\t\t Trainer is Fitting \t\t========")
  trainer.fit()
  print("========\t\t Trainer is evaluating \t\t========")
  trainer.evaluate()
  print("========\t\t Trainer is Wrapping Up \t\t========")
  print("🚀🚀🚀🚀🚀🚀\t\t Trainer Flow Completed! \t\t🚀🚀🚀🚀🚀🚀")