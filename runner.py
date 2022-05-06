import argparse
from email.policy import default
import os
from numpy import require
import tqdm

def runner(args):
  main_file = args.main_path
  python_type = args.python
  log_path =args.log_path
  configs_dir = args.configs
  run_type = args.run_type
  disabled_gpu = args.disabled_gpu
  gpu_used = args.gpu_used
  configs = [config for config in os.listdir(configs_dir)]
  configs.sort()
  print(f"Training {run_type} with config path {configs_dir} started!")
  for config_name in tqdm.tqdm(configs):
    print(f"Training with config {config_name} started")
    config_path = os.path.join(configs_dir, config_name)
    config_name_split = config_name.split(".")
    output_dir = os.path.join(log_path, f"{'.'.join(config_name_split[:len(config_name_split) - 1])}.txt")
    os.system(f'{python_type} "{main_file}" --config "{config_path}" --run_type "{run_type}" --disabled_gpu "{disabled_gpu}" --gpu_used "{gpu_used}" > "{output_dir}"')
  print(f"Training process completed!!")

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--configs", required=True, help="configs folder path"
  )
  parser.add_argument("--log_path", required=True, help="verbose output path")
  parser.add_argument("--main_path", required=True, help="file main path")
  parser.add_argument("--run_type", required=True, help="runner type [shallow || deep]")
  parser.add_argument("--python", required=True, help="[python3 || python]", default="python3")
  parser.add_argument("--disabled_gpu", required=False, help="GPU to disable in format 'GPU1 | GPU2'", default="")
  parser.add_argument("--gpu_used", required=False, help="GPU used 3 4 5")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_arguments()
  runner(args)