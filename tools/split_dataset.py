import os
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import argparse
from data_preprocessor import DataPreprocessor

class DataSplitter:
  def __init__(self, data_path, save_path, prefix = "data", is_sampling_enabled = False, test_size = 0.33, random_state=13518136, clean_null=True, preprocessing = False):
    if not os.path.isfile(data_path):
      raise ValueError("Data not found (incorrect data path)")
    if not os.path.isdir(save_path):
      raise ValueError("Save directory isn't found (incorrect save path)")
    self.data_path = data_path
    self.save_path = save_path
    self.prefix = prefix
    self.is_sampling_enabled = is_sampling_enabled
    self.test_size = test_size
    self.random_state = random_state
    self.clean_null = clean_null
    self.dataFrame = None
    self.ros_model = RandomOverSampler(random_state=self.random_state)
    self.rus_model = RandomUnderSampler(random_state=self.random_state)
    self.preprocessing = preprocessing
  
  def load_data(self):
    # Data must be csv
    self.dataFrame = pd.read_csv(self.data_path)
    print(f"Loaded data at {self.data_path}")
    total_null = sum(self.dataFrame.isnull().sum())
    if self.clean_null and total_null > 0:
      self.dataFrame.dropna(inplace=True)
      print(f"Total {total_null} row contains null values dropped")
    if self.preprocessing:
      self.dataFrame = DataPreprocessor(self.dataFrame, os.path.abspath(os.path.join(self.data_path, os.pardir))).main()

  def sampling(self, model_type = "ros", label_key = "Label"):
    """
    model_type ==> ros | rus
    """
    if model_type == "ros":
      SamplingModel = self.ros_model
    else:
      SamplingModel = self.rus_model
    X = self.dataFrame.drop([label_key], axis=1)
    Y = self.dataFrame[label_key]
    print(f"Resampling data with {model_type}")
    return SamplingModel.fit_resample(X, Y)
  
  def split(self, X, Y):
    return train_test_split(X, Y, random_state=self.random_state, test_size=self.test_size)

  def save_data(self, x1, x2, y1, y2, data_column_names=[], suffix="normal"):
    """
    suffix ==> normal | rus | ros
    """
    if len(data_column_names) == 0:
      data_column_names = self.dataFrame.columns.tolist()
    elif len(data_column_names != len(self.dataFrame.columns.tolist())):
      raise ValueError("data_column_names has different length with the data")
    train = pd.concat([x1, y1], axis=1)
    train.columns = data_column_names
    dev = pd.concat([x2, y2], axis=1)
    dev.columns = data_column_names
    # save data
    data_dir = os.path.join(self.save_path, f"{self.prefix}_{suffix}")
    os.mkdir(data_dir)
    train.to_csv(os.path.join(data_dir, "train.csv"))
    dev.to_csv(os.path.join(data_dir, "dev.csv"))
    print(f"New {self.prefix}_{suffix} data saved at {data_dir}")


  def main(self):
    self.load_data()
    # Normal
    X = self.dataFrame.drop(["Label"], axis=1)
    Y = self.dataFrame["Label"]
    self.save_data(*self.split(X, Y))
    # Sampling
    if self.is_sampling_enabled:
      # ros
      self.save_data(*self.split(*self.sampling()), suffix="ros")
      # rus
      self.save_data(*self.split(*self.sampling(model_type="rus")), suffix="rus")

if __name__  == '__main__':
  parser = argparse.ArgumentParser(
          description='Train Split test integrated with makedir os')
  parser.add_argument('--data_path', type=str, default='./dataset/data.csv',
                      help='Path to data to be splitted')
  parser.add_argument('--save_path', type=str, default='./dataset/split',
                      help='Path to output split data')
  parser.add_argument('--prefix', type=str, default='new',
                      help='Prefix name for new data folder')
  parser.add_argument('--is_sampling_enabled', type=bool, default=False,
                      help='Enable sampling')
  parser.add_argument('--test_size', type=float, default=0.33,
                      help='Test data Size')
  parser.add_argument('--random_state', type=int, default=13518136,
                      help='Test data Size')
  parser.add_argument('--clean_null', default=False, type=bool,
                      help='Clean null data')
  parser.add_argument('--preprocessing', default=False, type=bool,
                      help='Activate preprocessing with DataPreprocessing class')
  args = parser.parse_args()

  DataSplitter(data_path=args.data_path, save_path=args.save_path,
    prefix=args.prefix, is_sampling_enabled=args.is_sampling_enabled,
    test_size=args.test_size, random_state=args.random_state, 
    clean_null=args.clean_null, preprocessing=args.preprocessing).main()

  


