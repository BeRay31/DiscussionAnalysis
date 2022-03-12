import gensim
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd

class WordVectorsEmbedder:
  def __init__(self, config):
    self.config = config
    if self.config["model_type"] == "fasttext":
      self.EmbedderClass = gensim.models.fasttext.FastText
    else:
      self.EmbedderClass = gensim.models.word2vec.Word2Vec
      
    if self.config["is_pretrained"]:
      self.model = self.EmbedderClass.load(self.config["model_path"])
    
    if self.config["log_oov"]:
      self.oov_log = []
    self.padding = np.zeros(self.config["vector_size"],)

  def get_word_vector(self, word):
    try:
      word_vector = self.model.wv.get_vector(word)
    except KeyError:
      word_vector = self.padding
      if self.enable_oov_log:
        self.oov_log.append(word)
    return word_vector

  def text_to_vectors(self, text):
    if not bool(self.model):
      raise ValueError("There is no model loaded")
    tokens = word_tokenize(text)
    # Normalize the final vector based on sentence length
    vector_length = len(tokens) if len(tokens) < self.config["max_sequence"] else self.config["max_sequence"]
    vector = []
    for token in tokens[:vector_length]:
      vector.append(self.get_word_vector(token))
    
    pad = self.config["max_sequence"] - len(vector)
    for i in range(pad):
      vector.append(self.padding)
    
    final_vector = np.asarray(vector).flatten() 
    return final_vector

  def df_to_vector(self, df, concat = True):
    key_list = self.config["key_list"].split("_")
    vector = []
    is_concat = self.config["model_behavior"] == "concat"
    for idx, row in df.iterrows():
      vector_temp = []
      for key in keyList:
        temp = self.text_to_vector(row[key])
        if is_concat:
          temp = temp.tolist()
        if len(vector_temp) == 0:
          vector_temp = temp
        else:
          vector_temp += temp
      vector.append(vector_temp)
    vector = np.asarray(vector)
    res = vector
    return res

  def fit(self, data):
    key_list = self.config["key_list"].split("_")
    sentences = []
    for key in key_list:
      sentences += data[key].unique().tolist()
    tokenized_sentences = [word_tokenize(sent) for sent in sentences]
    if not self.model:
      self.model = self.EmbedderClass(tokenized_sentences, vector_size=self.config["vector_size"],
        window=self.config["window"], min_count=self.config["window"], epochs=self.config["epoch"],
        seed=13518136)
    else:
      self.model.build_vocab(tokenized_sentences)
      self.model.train(tokenized_sentences, total_examples=self.model.corpus_count, epochs=self.model.epochs)

      

