import os
import yaml
import time
from datetime import datetime
import pickle

def save_model_with_pickle(model, path):
  pickle_out = open(path, "wb")
  pickle.dump(model, pickle_out)
  pickle_out.close()

def get_config(path):
  return yaml.safe_load(open(path, "r"))

def construct_datetime(time):
  return datetime.fromtimestamp(time).strftime("%A, %B %d, %Y %I:%M:%S")

def construct_time(time_start, time_end = 0):
  if time_end == 0:
    total = time_start
  else:
    total = time_end - time_start
  return time.strftime('%H : %M : %S', time.gmtime(total))

def dump_config(path, config):
  with open(path, "w+") as f:
    f.write(yaml.dump(config))

def get_latest_version(folder, prefix, mode="patch"):
  """
  mode --> (major | minor | patch)
  """
  files = os.listdir(folder)
  versions = []
  for file in files:
    if file.startswith(prefix):
      version = [int(ver) for ver in file.split("-")[-1].split(".")]
      versions.append(version)

  if not versions:
    return "0.0.0"

  major, minor, patch = sorted(versions, reverse=True)[0]
  if mode == "major":
    return f"{major+1}.0.0"
  elif mode == "minor":
    return f"{major}.{minor+1}.0"
  elif mode == "patch":
    return f"{major}.{minor}.{patch+1}"
  else:
    raise ValueError("only support (major | minor | patch) => (major).(minor).(patch")