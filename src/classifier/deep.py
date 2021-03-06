import tensorflow as tf
from transformers import TFBertModel, TFXLMRobertaModel

tf.random.set_seed(13518136)
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    GRU,
    Dropout,
    Bidirectional
)
WORD_VECTORS = ["word2vec", "fasttext"]
class DeepClassifier(Model):
  def __init__(self, config):
    super().__init__()
    self.config = config

    # Define Embedding Layer
    if self.config["type"] == "bert":
      self.embedder = TFBertModel.from_pretrained(config["model_name"])
    elif config["type"].lower() == "roberta":
      self.embedder = TFXLMRobertaModel.from_pretrained(config["model_name"])
    
    # Define Recurrent Layer
    if self.config["recurrent_layer"].lower() == "lstm":
      self.recurrent = LSTM(self.config["recurrent_unit"])
    elif self.config["recurrent_layer"].lower() == "bilstm":
      self.recurrent = Bidirectional(LSTM(self.config["recurrent_unit"]))
    elif self.config["recurrent_layer"].lower() == "gru":
      self.recurrent = GRU(self.config["recurrent_unit"])
    elif self.config["recurrent_layer"].lower() == "bigru":
      self.recurrent = Bidirectional(GRU(self.config["recurrent_unit"]))
    elif self.config["recurrent_layer"].lower() == "dense":
      self.recurrent = Dense(self.config["recurrent_unit"], activation="relu", name="dense")
    else:
      self.recurrent = None
      raise ValueError("only support (lstm | bilstm | gru | bigru) layer type")
    
    # Define features dropout
    self.dropout = Dropout(self.config["recurrent_dropout"])
    labels = self.config["labels"].split("_")
    
    # Define Output Layer
    self.out = Dense(len(labels), activation="softmax", name="classifier")
  
  def call(self, X, training=None):
    if not self.config["type"] in WORD_VECTORS:
      # Pass to BERT or XLNet Model
      if self.config["recurrent_layer"].lower() == "dense":
        X_embed = self.embedder(X)["pooler_output"]
      else:
        X_embed = self.embedder(X)["last_hidden_state"]
    else:
      X_embed = X
    
    # Passed to recurrent layer
    X_recurrent = X_embed
    if self.recurrent != None:
      X_recurrent = self.recurrent(X_embed)
    
    # Dropout features to decrease overfit
    X_ = self.dropout(X_recurrent, training=training)
    
    # Final Dense layer
    res = self.out(X_)

    return res
    

