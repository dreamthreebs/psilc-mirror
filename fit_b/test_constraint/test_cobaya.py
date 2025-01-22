from cobaya.yaml import yaml_load_file
from cobaya.run import run

info = yaml_load_file("gaussian.yaml")
updated_info, sampler = run(info)

