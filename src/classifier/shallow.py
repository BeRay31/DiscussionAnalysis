from sklearn.svm import SVC
from sklearn.decomposition import PCA
from src.embedder import WordVectorsEmbedder
from src.util import load_model_with_pickle
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import time
import pandas as pd
class ShallowClassifier:
  def __init__(self, config):
    self.config = config
    self.model = None
    self.decomposer = None
    self.train_decomposer_time = None
    self.train_model_time = None
    self.overall_predict_time = None
    self.model_score = None
    self.confusion_matrix = None
    self.classification_report = None
    self.pred = None
    if self.config["load_model"]:
      self.model = load_model_with_pickle(self.config["model_path"])
    if self.config["decomposer"]["load_model"]:
      self.decomposer = load_model_with_pickle(self.config["decomposer"]["model_paths"])

  def fit_decomposer(self, X):
    start = time.time()
    self.decomposer = PCA(**self.config["decomposer"]["decomposer_config"], random_state=13518136)
    self.decomposer.fit(X)
    end = time.time()
    self.train_decomposer_time = round(end - start, 2)
  
  def fit_model(self, X, Y):
    start = time.time()
    self.model = SVC(**self.config["svm_config"], random_state=13518136)
    self.model.fit(X, Y)
    end = time.time()
    self.train_model_time = round(end - start, 2)
  
  def fit(self, data, embedder: WordVectorsEmbedder):
    X = embedder.df_to_vector(data)
    Y = data[self.config["label_key"]]
    
    # Train PCA
    self.fit_decomposer(X)
    X = self.decomposer.transform(X)

    # Train Model
    self.fit_model(X, Y)
  
  def evaluate(self, data, embedder):
    if not self.model:
      raise ValueError("Model doesn't available")
    start = time.time()

    # Transform data
    X = embedder.df_to_vector(data)
    Y = data[self.config["label_key"]]
    X = self.decomposer.transform(X)
    labels = self.config["labels"].split("_")
    # Recap and predict
    pred = self.model.predict(X)
    end = time.time()
    overall_predict_time = round(end - start, 2)
    model_score = self.model.score(X, Y)
    model_f1_score = f1_score(
        y_true=Y,
        y_pred=pd.DataFrame(pred),
        labels=labels,
        average='weighted'
      )
    # Metrics
    confusion_mat = confusion_matrix(Y, pred, labels=labels)
    classification_rep = classification_report(Y, pred, labels=labels)
    pred = pd.DataFrame(pred)
    pred = pd.concat([data, pred], axis=1)
    pred.columns = data.columns.tolist() + ["Prediction"]

    return {
      "predict_time": overall_predict_time,
      "score": model_score,
      "f1_score": model_f1_score,
      "confusion_matrix": confusion_mat,
      "classification_report": classification_rep,
      "prediction": pred
    }
