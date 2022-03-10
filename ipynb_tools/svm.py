from datetime import datetime
import os
import pandas as pd
from ipynb_tools.fasttext import FastText

from ipynb_tools.w2v import Word2Vec
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import time
import pickle

class Trainer:
  def __init__(self, embedding = "w2v", embedding_path = None, embedding_behavior = "concat", save_directory = "./", data_dir = "./", prefix = "train", version_mode = "patch"):
    """
    version_mode --> (major | minor | patch)
    embedding --> "w2v | fasttext"
    """
    self.save_directory = save_directory
    self.data_dir = data_dir
    self.prefix = prefix
    self.version_mode = version_mode
    self.embedding = embedding
    self.embedding_path = embedding_path
    self.embedding_behavior = embedding_behavior
    self.embedder = None
    self.pca_model = None
    # Metrics
    self.train_embedder_time = None
    self.train_pca_time = None
    self.train_svc_time = None
    self.overall_predict_time = None
    self.confusion_matrix = None
    self.classification_report = None
    self.score = None
    self.res_data = None
    self.could_log_metrics = False
    self.__init_folder()
  
  def get_latest_version(self):
    files = os.listdir(self.save_directory)
    versions = []
    for file in files:
      if file.startswith(self.prefix):
        version = [int(ver) for ver in file.split("-")[-1].split(".")]
        versions.append(version)
    
    if not versions:
      return "0.0.0"
    major, minor, patch = sorted(versions, reverse=True)[0]
    if self.version_mode == "major":
      return f"{major+1}.0.0"
    elif self.version_mode == "minor":
      return f"{major}.{minor+1}.0"
    elif self.version_mode == "patch":
      return f"{major}.{minor}.{patch+1}"
    else:
      raise ValueError("only support (major | minor | patch) => (major).(minor).(patch")

  def __init_folder(self):
    version = self.get_latest_version()
    self.res_dir_name = os.path.join(self.save_directory, f"{self.prefix}-{version}")
    os.mkdir(self.res_dir_name)

  def load_data(self, name):
    data_path = os.path.join(self.data_dir, name)
    if not os.path.exists(data_path):
      raise ValueError("Data doesn't exist")
    df = pd.read_csv(data_path)
    return df

  def get_embedding_model(self, data, key_list = []):
    if self.embedding == "w2v":
      Embedder = Word2Vec
    elif self.embedding == "fasttext":
      Embedder = FastText
    else:
      raise ValueError("Embedding available => [ w2v | fasttext ]")
    if self.embedding_path:
      self.embedder = Embedder(load_model=True, model_path=self.embedding_path)
    elif not data.empty:
      start = time.time()
      self.embedder = Embedder()
      self.embedder.fit(data, key_list=key_list)
      end = time.time()
      self.train_embedder_time = round(end - start, 2)
    return self.embedder
  
  def get_pca_model(self, data):
    start = time.time()
    self.pca_model = PCA(0.8, random_state=13518136)
    self.pca_model.fit(data)
    end = time.time()
    self.train_pca_time = round(end - start, 2)
  
  def fit(self):
    # load train data
    data = self.load_data("train.csv")
    key_list = ["Tweet", "Comment"]
    # get feature extraction model
    self.get_embedding_model(data, key_list)
    
    # transform vector
    is_concat = self.embedding_behavior == "concat"
    x_train = self.embedder.df_to_vector(data, key_list, concat=is_concat)
    y_train = data["Label"]

    # PCA model
    self.get_pca_model(x_train)
    # transform wih pca
    x_train = self.pca_model.transform(x_train)

    # initiate model
    start = time.time()
    self.model = SVC(random_state=13518136)
    self.model.fit(x_train, y_train)
    end = time.time()
    self.train_svc_time = round(end - start, 2)
  
  def predictAndEvaluate(self):
    if not self.model:
      raise ValueError("Model doesn't available")
    # load test data
    data = self.load_data("test.csv")
    key_list = ["Tweet", "Comment"]
  
    start = time.time()
    # transform vector
    is_concat = self.embedding_behavior == "concat"
    x_test = self.embedder.df_to_vector(data, key_list, concat=is_concat)
    y_test = data["Label"]

    # transform wih pca
    x_test = self.pca_model.transform(x_test)
    
    # get score
    self.score = self.model.score(x_test, y_test)
    pred = self.model.predict(x_test)
    end = time.time()
    self.overall_predict_time = round(end - start, 2)
    
    self.confusion_matrix = confusion_matrix(y_test, pred, labels=['Uncorrelated', 'Contra Sarcasm', 'Pro', 'Neutral', 'Contra', 'Pro Sarcasm'])
    self.classification_report = classification_report(y_test, pred, labels=['Uncorrelated', 'Contra Sarcasm', 'Pro', 'Neutral', 'Contra', 'Pro Sarcasm'])
    x = pd.concat([data["Tweet"], data["Comment"]], axis=1)
    y_test = pd.DataFrame(y_test)
    pred = pd.DataFrame(pred)
    self.res_data = pd.concat([x, y_test, pred], axis=1)
    self.res_data.columns = key_list + ["Labels", "Predictions"]
    self.could_log_metrics = True

  def save_model_with_pickle(self, model, path):
    pickle_out = open(path, "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()

  def save_and_log_metrics(self):
    # Save SVC Model
    if self.model:
      self.save_model_with_pickle(self.model, os.path.join(self.res_dir_name, "classifier.model"))
    # Save PCA Model
    if self.pca_model != None:
      self.save_model_with_pickle(self.pca_model, os.path.join(self.res_dir_name, "pca.model"))
    # Save Embedder Model
    if self.embedder != None:
      self.embedder.model.save(os.path.join(self.res_dir_name, "embedder.model"))
    # Save Prediction to CSV
    if not self.res_data.empty:
      self.res_data.to_csv(os.path.join(self.res_dir_name, "prediction.csv"))
    # Log Metrics
    if self.could_log_metrics:
      with open(os.path.join(self.res_dir_name, "log.txt"), "w+") as f:
        msg = f"Experiment datetime: {datetime.now()}\n"
        msg += f"Experiment Prefix: {self.prefix}\n"
        msg += f"Data Path: {self.data_dir}\n"
        msg += "\n"
        msg += "====\t\t Embedder Details \t\t====\n\n"
        msg += f"Embedding Path: {self.embedding_path}\n"
        msg += f"Embedding Type: {self.embedding}\n"
        msg += f"Embedding Behavior: {self.embedding_behavior}\n"
        msg += f"Embedding vocab length: {len(self.embedder.model.wv)}\n"
        msg += f"Embedding vector length: {self.embedder.model.wv.vector_size}\n"
        msg += f"Embedding window: {self.embedder.model.window}\n"
        msg += f"Embedding total train time: {self.embedder.model.total_train_time}\n"
        msg += f"Embedding total train count: {self.embedder.model.train_count}\n"
        msg += f"Current train time: {self.train_embedder_time}\n"
        msg += "\n"
        msg += "====\t\t PCA Details \t\t====\n\n"
        msg += f"Current train time: {self.train_pca_time}\n"
        msg += f"Number of Features: {self.pca_model.n_features_}\n"
        msg += f"Number of Components: {self.pca_model.n_components_}\n"
        msg += f"Total explained variance ratio: {sum(self.pca_model.explained_variance_ratio_)}\n"
        msg += f"Noise Variance: {self.pca_model.noise_variance_}\n"
        msg += f"Total pca mean: {sum(self.pca_model.mean_)}\n"
        msg += "\n"
        msg += "====\t\tClassification Details\t\t====\n\n"
        msg += f"Current Train Time: {self.train_svc_time}\n"
        msg += f"Overall Predict Time: {self.overall_predict_time}\n"
        msg += f"Classifier Score: {self.score}\n"
        msg += f"Labels:\n{['Uncorrelated', 'Contra Sarcasm', 'Pro', 'Neutral', 'Contra', 'Pro Sarcasm']}\n"
        msg += f"\nConfusion matrix:\n{self.confusion_matrix}\n"
        msg += f"\nClassification_ Report:\n{self.classification_report}\n"
        f.write(msg)

  def main(self):
    self.fit()
    self.predictAndEvaluate()
    self.save_and_log_metrics()
    self.save_and_log_metrics()

