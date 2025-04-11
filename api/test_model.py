import mlflow
from torch import Size
from torchinfo import summary
import os
import dotenv
env_var = dotenv.load_dotenv("app_config.env")
if env_var:
    pass
else:
    raise FileNotFoundError("Application environment variables file not found.")


model_uri = os.getenv("MODEL_URI")
# model_uri = "../mlartifacts/368316252121826279/70d9c885e3e84885b91585640a06084f/artifacts/models"


model = mlflow.pytorch.load_model(model_uri, map_location="cpu")

total_parameters = sum([p.numel() for p in model.parameters()])
input_layer_size = model.token_embd_layer.weight.shape
# model_summary = summary(model)
# print(model_summary)
# print(type(model_summary))
# # print(model)

def test_num_parameters():
    assert total_parameters == 35265536

def test_input_embd_shape():
    assert input_layer_size == Size([50257, 256])