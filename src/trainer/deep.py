from src.loader import DeepLoader
from src.trainer import Trainer
from src.classifier import DeepClassifier
from src.callback import CustomSaver

import tensorflow as tf
import os
import numpy as np
import pandas as pd
from time import time

tf.random.set_seed(13518136)
from src.util import construct_time, dump_config
from datetime import datetime

from tensorflow.keras.callbacks import (
  LearningRateScheduler
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score


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
    self.strategy = None
    # Set GPU
    gpus = tf.config.experimental.list_physical_devices("GPU")
    gpu_names = [gpu.name.split("e:")[1] for gpu in gpus]
    config_taken = set([int(i) for i in self.config["trainer"]["gpus"].split("|")])
    taken_gpu = []
    for taken in config_taken:
      for gpu_name in gpu_names:
        if str(taken) in gpu_name:
          taken_gpu.append(gpu_name)
    print(f"Taken GPU: {taken_gpu}")
    self.strategy = tf.distribute.MirroredStrategy(devices=taken_gpu)

  # Convert arr to label
  def get_label(self, arr):
    labels = self.config["master"]["labels"].split("_")
    idx = np.argmax(arr)
    return labels[idx]

  def fit(self):
    # mkdir for models
    self.models_path = os.path.join(self.directory_path, "models")
    os.mkdir(self.models_path)

    with self.strategy.scope():
      self.model = DeepClassifier(
        {**self.config["master"], **self.config["classifier"]}
      )
      start = time()

      # Train with Freeze embedding weights
      if self.config["trainer"]["freeze_embedding"]:
        self.model.embedder.trainable = False
        # Didn't save the model cuz it doesn't train the whole model
        freeze_callbacks = [
          LearningRateScheduler(scheduler)
        ]
        freeze_embedder_config = self.config["trainer"]["train_freeze"]
        # Compile freeze model
        self.model.compile(
          loss=CategoricalCrossentropy(),
          optimizer=Adam(learning_rate=freeze_embedder_config["learning_rate"]),
        )
        # fit freeze model
        self.model.fit(
          self.data["x_train"],
          self.data["y_train"],
          batch_size=freeze_embedder_config["batch_size"],
          epochs=freeze_embedder_config["epochs"],
          validation_data=(self.data["x_dev"], self.data["y_dev"]),
          callbacks=freeze_callbacks,
        )
        self.model.summary()
        self.model.embedder.trainable = True

      # Compile model
      self.model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(learning_rate=self.config["trainer"]["learning_rate"]),
      )

      # Define train callbacks
      callbacks = [
        LearningRateScheduler(scheduler),
        CustomSaver(self.models_path)
      ]

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

  def evaluate(self):
    # prepare dir for evaluation
    self.eval_path = os.path.join(self.directory_path, "evaluation")
    os.mkdir(self.eval_path)

    model_weights = [weight for weight in os.listdir(self.models_path)]
    for model_weight in model_weights:
      # Load best model
      self.model.load_weights(os.path.join(self.models_path, model_weight))
      
      with self.strategy.scope():
        start = time()
        pred_train = self.model.predict(
          self.data["x_train"],
          batch_size=self.config["trainer"]["batch_size"],
          verbose=1,
        )
        end = time()
        train_pred_time = round(end - start, 2)

        start = end
        pred_dev = self.model.predict(
          self.data["x_dev"],
          batch_size=self.config["trainer"]["batch_size"],
          verbose=1,
        )
        end = time()
        dev_pred_time = round(end - start, 2)

        test_pred_time = None
        pred_test = np.asarray([])
        if not self.data["test"].empty:
          start = end
          pred_test = self.model.predict(
            self.data["x_test"],
            batch_size=self.config["trainer"]["batch_size"],
            verbose=1
          )
          end = time()
          test_pred_time = round(end - start, 2)

        self.save(pred_train, pred_dev, pred_test, train_pred_time, dev_pred_time, test_pred_time, model_weight)

  def save(self, y_train_pred: np.array, y_dev_pred: np.array, y_test_pred: np.array, train_pred_time, dev_pred_time, test_pred_time, model_name):
    train_key = self.config["master"]["train_key"]
    dev_key = self.config["master"]["dev_key"]
    test_key = self.config["master"]["test_key"]
    label_key = self.config["master"]["label_key"]
    # Prepare Folder
    model_names = model_name.split(".")
    model_name = ".".join(model_names[:len(model_names) - 1])
    save_path = os.path.join(self.eval_path, model_name)
    os.makedirs(save_path) 

    labels = self.config["master"]["labels"].split("_")
    # Save CSV
    train_prediction = [self.get_label(arr) for arr in y_train_pred]
    train_prediction = pd.DataFrame(train_prediction, columns=["Prediction"])
    df_train = pd.concat([self.data[train_key], train_prediction], axis = 1)
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
    train_accuracy_score = accuracy_score(
      y_true=df_train[self.config["master"]["label_key"]],
      y_pred=df_train["Prediction"]
    )
    train_f1_score = f1_score(
      y_true=df_train[self.config["master"]["label_key"]],
      y_pred=df_train["Prediction"],
      labels=labels,
      average='weighted'
    )

    dev_prediction = [self.get_label(arr) for arr in y_dev_pred]
    dev_prediction = pd.DataFrame(dev_prediction, columns=["Prediction"])
    df_dev = pd.concat([self.data[dev_key], dev_prediction], axis = 1)
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
    dev_accuracy_score = accuracy_score(
      y_true=df_dev[self.config["master"]["label_key"]],
      y_pred=df_dev["Prediction"]
    )
    dev_f1_score = f1_score(
      y_true=df_dev[self.config["master"]["label_key"]],
      y_pred=df_dev["Prediction"],
      labels=labels,
      average='weighted'
    )

    # Save to CSV
    df_train.to_csv(os.path.join(save_path, "pred_train.csv"), index=False)
    df_dev.to_csv(os.path.join(save_path, "pred_dev.csv"), index=False)

    test_confusion_matrix = None
    test_classification_report = None
    test_accuracy_score = None
    if not self.data[test_key].empty:
      test_prediction = [self.get_label(arr) for arr in y_test_pred]
      test_prediction = pd.DataFrame(test_prediction, columns=["Prediction"])
      df_test = pd.concat([self.data[test_key], test_prediction], axis = 1)
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
      test_accuracy_score = accuracy_score(
        y_true=df_test[self.config["master"]["label_key"]],
        y_pred=df_test["Prediction"]
      )
      test_f1_score = f1_score(
        y_true=df_test[self.config["master"]["label_key"]],
        y_pred=df_test["Prediction"],
        labels=labels,
        average='weighted'
      )

    # Data distribution
    train_data_distribution = self.data[train_key][f"{label_key}"].value_counts()
    dev_data_distribution = self.data[dev_key][f"{label_key}"].value_counts()
    test_data_distribution = None
    if not self.data[test_key].empty:
      test_data_distribution = self.data[test_key][f"{label_key}"].value_counts()

    # Log Result
    with open(os.path.join(save_path, "log.txt"), "w+") as f:
      msg = "Experiment datetime: {}\n".format(datetime.now())
      msg += "Experiment prefix: {}\n".format(self.config["master"]["prefix"])
      msg += "Experiment description: {}\n".format(self.config["master"]["description"])
      msg += "Data path: {}\n".format(self.config["loader"]["data_path"])
      msg += "\n"
      msg += "========\t\t Data Train Details \t\t========\n"
      msg += "\n"
      msg += "Data train length: {}\n".format(len(self.data[train_key]))
      msg += "Data train distributions: \n{}\n".format(train_data_distribution)
      msg += "\n"
      msg += "========\t\t Data Dev Details \t\t========\n"
      msg += "\n"
      msg += "Data dev length: {}\n".format(len(self.data[dev_key]))
      msg += "Data dev distributions: \n{}\n".format(dev_data_distribution)
      msg += "\n"
      msg += "========\t\t Data Test Details \t\t========\n"
      msg += "\n"
      msg += "Data test length: {}\n".format(len(self.data[test_key]))
      msg += "Data test distributions: \n{}\n".format(test_data_distribution)
      msg += "\n"
      msg += "\n"
      msg += "========\t\t Trainer Details \t\t========\n"
      msg += "Is sampling enabled: {}\n".format(self.config["master"]["sampling"])
      msg += "Training batch size: {}\n".format(self.config["trainer"]["batch_size"])
      msg += "Training learning rate: {}\n".format(self.config["trainer"]["learning_rate"])
      msg += "Training Epochs: {}\n".format(self.config["trainer"]["epochs"])
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
      msg += "========\t\t Training Prediction Metrics Details Recap \t\t========\n\n"
      msg += "\nAccuracy:\n{}\n".format(train_accuracy_score)
      msg += "\nAccuracy:\n{}\n".format(train_f1_score)
      msg += "\nConfusion matrix:\n{}\n".format(train_confusion_matrix)
      msg += "\nClassification report:\n{}\n".format(train_classification_report)
      msg += "\n"
      msg += "========\t\t Dev Prediction Metrics Details Recap \t\t========\n\n"
      msg += "\nAccuracy:\n{}\n".format(dev_accuracy_score)
      msg += "\nAccuracy:\n{}\n".format(dev_f1_score)
      msg += "\nConfusion matrix:\n{}\n".format(dev_confusion_matrix)
      msg += "\nClassification report:\n{}\n".format(dev_classification_report)
      msg += "\n"
      msg += "========\t\t test Prediction Metrics Details Recap \t\t========\n\n"
      msg += "\nAccuracy:\n{}\n".format(test_accuracy_score)
      msg += "\nAccuracy:\n{}\n".format(test_f1_score)
      msg += "\nConfusion matrix:\n{}\n".format(test_confusion_matrix)
      msg += "\nClassification report:\n{}\n".format(test_classification_report)
      msg += "\n"
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
  print("========\t\t Trainer Flow Completed! \t\t========")