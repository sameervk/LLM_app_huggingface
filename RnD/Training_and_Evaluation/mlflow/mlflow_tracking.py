import os
import dotenv
import yaml

import mlflow

from RnD.LLM_arch.GPT2.llm_gpt2 import GPT2

dotenv.load_dotenv("../mlflow_config.env")

uri_host = os.getenv("HOST")+":"+os.getenv("PORT")
print(f"Tracking server: {uri_host}")

# set tracking uri
mlflow.set_tracking_uri(uri=uri_host)

# set experiment
experiment = mlflow.set_experiment(experiment_name="GPT2-smalldataset-pytorch")

# add any tags to the experiment
experiment_tags = {
    "description": "Training of GPT2 architecture-based LLM using Pytorch",
    "data": "The Verdict from https://en.wikisource.org/wiki/The_Verdict"
}
mlflow.set_experiment_tags(tags=experiment_tags)

# Initiate LLM
## import config
with open("../llm_config.yaml", "r") as file:
    llm_config = yaml.safe_load(file.read())

model = GPT2(**llm_config)

# import training parameters
with open("../training_parameters.yaml", "r") as file:
    train_params = yaml.safe_load(file.read())

with mlflow.start_run():
    mlflow.log_params()