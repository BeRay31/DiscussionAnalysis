import gensim
import multiprocessing
import os.path
import requests
import argparse
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import sys
import time
from datetime import datetime
from gensim.models.callbacks import CallbackAny2Vec

class TrainCallbacks(CallbackAny2Vec):
  def __init__(self):
    self.epoch = 0
    self.train_started_at = None
    self.train_end_at = None
    self.epoch_started_at = None
    self.epoch_end_at = None

  def construct_time(self, time_start, time_end):
    total = time_end - time_start
    return time.strftime('%H : %M : %S', time.gmtime(total))
  
  def construct_datetime(self, time):
    return datetime.fromtimestamp(time).strftime("%A, %B %d, %Y %I:%M:%S")

  def on_epoch_begin(self, model):
    self.epoch += 1
    self.epoch_started_at = time.time()
    print(f"Epoch {self.epoch} started at {self.construct_datetime(self.epoch_started_at)}")
  
  def on_epoch_end(self, model):
    self.epoch_end_at = time.time()
    print(f"Epoch {self.epoch} ends at {self.construct_datetime(self.epoch_end_at)}")
    print(f"Epoch time: {self.construct_time(self.epoch_started_at, self.epoch_end_at)}")
  
  def on_train_begin(self, model):
    self.train_started_at = time.time()
    print(f"Training started at {self.construct_datetime(self.train_started_at)}")

  def on_train_end(self, model):
    self.train_end_at = time.time()
    print(f"Training ended at {self.construct_datetime(self.train_end_at)}")
    print(f"Total training time: {self.construct_time(self.train_started_at, self.train_end_at)}")
        
    

def download(link, file_name):
  with open(file_name, "wb") as f:
    print("Downloading %s" % file_name)
    response = requests.get(link, stream=True)
    total_length = response.headers.get('content-length')

    if total_length is None: # no content length header
      f.write(response.content)
    else:
      dl = 0
      total_length = int(total_length)
      for data in response.iter_content(chunk_size=4096):
        dl += len(data)
        f.write(data)
        done = int(50 * dl / total_length)
        sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )
        sys.stdout.flush()

def get_id_wiki(dump_path):
  if not os.path.isfile(dump_path):
    url = 'https://dumps.wikimedia.org/idwiki/latest/idwiki-latest-pages-articles.xml.bz2'
    download(url, dump_path)
  return gensim.corpora.WikiCorpus(dump_path, dictionary={})

def extract_text(extracted_path, id_wiki, stem):
  if os.path.isfile(extracted_path):
    return None
  if stem:
    print('Warning : Using stemmer could slow down the extracting progress')
    stemmer = StemmerFactory().create_stemmer()
  start = time.time()
  with open(extracted_path, 'w', encoding="utf-8") as f:
    i = 0
    for text in id_wiki.get_texts():
      text = ' '.join(text)
      text = stemmer.stem(text) if stem else text
      if str(text):
        f.write(text + '\n')
        i += 1
      if i%(10 if stem else 1000) == 0:
        print(str(i), 'articles processed')
    print('total:', str(i))
  end = time.time()
  print(f"extract time elpased time: {round(end-start,2)}")
  return None

def build_model(extracted_path, model_path, dim, model_type: str):
  sentences = gensim.models.word2vec.LineSentence(extracted_path)
  ModelClass = None
  if model_type.lower() == 'fasttext':
    print("Training FastText Model")
    ModelClass = gensim.models.fasttext.FastText
  else:
    print("Training Word2Vec Model")
    ModelClass = gensim.models.word2vec.Word2Vec
  print("Train Started")
  model = ModelClass(sentences, vector_size=dim, workers=multiprocessing.cpu_count()-1, callbacks=[TrainCallbacks()])
  model.callbacks = ()
  model.save(model_path)
  return model

def main(args):
  model_path = args.model_path
  model_type = args.model_type
  if model_type.lower() not in ['fasttext', 'word2vec']:
    raise ValueError("model_type must be fasttext | word2vec")
  extracted_path = args.extracted_path
  dump_path = args.dump_path
  dim = args.dim
  stem = args.stem
  id_wiki = get_id_wiki(dump_path)
  print('Extracting text...')
  extract_text(extracted_path, id_wiki, stem)
  print('Build a model...')
  build_model(extracted_path, model_path, dim, model_type)
  print('Saved model:', model_path)

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
          description='Generating FastText or Word2Vec model for bahasa Indonesia')
  parser.add_argument('--model_path', type=str, default='./model/word2vec.model',
                      help='path for saving trained models')
  parser.add_argument('--model_type', type=str, default='word2vec',
                      help='model type fasttext | word2vec')
  parser.add_argument('--extracted_path', type=str, default='./data/idwiki.txt',
                      help='path for extracting text')
  parser.add_argument('--dump_path', type=str, default='./data/idwiki-latest-pages-articles.xml.bz2',
                      help='path for dump data')
  parser.add_argument('--dim', type=int, default=100,
                      help='embedding size')
  parser.add_argument('--stem', default=False, type=lambda x: (str(x).lower() == 'true'),
                      help='use stemmer or not. (default false)')
  args = parser.parse_args()
  main(args)
