import os
import yaml

def get_config(path):
  return yaml.safe_load(open(path, "r"))

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