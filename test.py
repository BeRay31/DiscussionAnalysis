from src.wrapper import Wrapper
import pandas as pd


def correlation(x):
  if x["Label"] == "Uncorrelated":
    x["Correlation"] = "Uncorrelated"
  else:
    x["Correlation"] = "Correlated"
  return x

if __name__ == '__main__':
  model_wrapper = Wrapper(config_path="D:\\WorkBench\\TA NLP\\test.yml")
  df_test = pd.read_csv("D:\\WorkBench\\TA NLP\\res_deep\\res_53\\4_label\\pred_test.csv")
  df_test.drop(["Prediction"], axis=1, inplace=True)
  df_test = df_test.apply(correlation, axis=1)
  pred = model_wrapper.predict_2Pipeline_df(df_test)