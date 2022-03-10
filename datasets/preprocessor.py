from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import preprocessor as tp
tp.set_options(tp.OPT.URL, tp.OPT.MENTION, tp.OPT.URL, tp.OPT.EMOJI, tp.OPT.SMILEY)
import pandas as pd
from tqdm import tqdm

def preprocess_tweet(tweet):
  tempTweet = tp.tokenize(tweet)
  tempTweet = re.sub(r'[^\w\s]', '', tempTweet)
  stemmer = StemmerFactory().create_stemmer()
  tempTweet = stemmer.stem(tempTweet)
  return tempTweet

def preprocess_data(data_path, save_path):
  dataset = pd.read_csv(data_path)
  newHeader = {
    "Tweet": "Tweet",
    "Quote Tweet | Reply": "Comment",
    "Tweet Domain": "Domain",
    "Reyvan": "Label"
  }
  renamedDs = dataset.rename(columns=newHeader)
  for idx, row in tqdm(renamedDs.iterrows(), total=renamedDs.shape[0]):
    renamedDs.loc[idx, "Tweet"] = preprocess_tweet(renamedDs.loc[idx, "Tweet"])
    renamedDs.loc[idx, "Comment"] = preprocess_tweet(renamedDs.loc[idx, "Comment"])
  renamedDs = renamedDs.dropna(inplace=True)
  renamedDs.to_csv(save_path)

preprocess_data("D:\WorkBench\TA NLP\dataset_rey.csv", "D:\WorkBench\TA NLP\dataset_rey(preprocessed).csv")