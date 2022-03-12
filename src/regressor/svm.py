from sklearn.svm import SVC
from sklearn.decomposition import PCA
from src.embedder import WordVectorsEmbedder
from sklearn.metrics import confusion_matrix, classification_report
import time
import pandas as pd
class SVMRegressor:
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


  def fit_decomposer(self, X):
    start = time.time()
    self.decomposer = PCA(n_components=self.config["decomposer"]["variance_tolerance"], svd_solver=self.config["decomposer"]["svd_solver"],
      random_state=13518136)
    self.decomposer.fit(X)
    end = time.time()
    self.train_decomposer_time = round(end - start, 2)
  
  def fit_model(self, X, Y):
    start = time.time()
    self.model = SVC(kernel=self.config["kernel"], gamma=self.config["gamma"],
      max_iter=self.config["max_iter"], degree=self.config["degree"], verbose=self.config["verbose"], random_state=13518136)
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

    # Recap and predict
    pred = self.model.predict(X)
    end = time.time()
    self.overall_predict_time = round(end - start, 2)
    self.model_score = self.model.score(X, Y)
    # Metrics
    labels = self.config["labels"].split("_")
    self.confusion_matrix = confusion_matrix(Y, pred, labels=labels)
    self.classification_report = classification_report(Y, pred, labels=labels)

    pred = pd.DataFrame(pred)
    self.pred = pd.concat([data, pred], axis=1)
    self.pred.columns = data.columns.tolist() + ["Prediction"]
