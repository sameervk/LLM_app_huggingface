import os
import dotenv
import yaml
import argparse
from pathlib import Path
import mlflow

import torch
from tokenizers import Tokenizer
from torchmetrics.text import Perplexity
from torchinfo import summary
from math import exp

# from RnD.LLM_arch.GPT2.llm_gpt2 import GPT2
from RnD.Dataloader.dataloader import GPTDatasetV1, create_dataloader_v1
from RnD.Training_and_Evaluation.pytorch.utils import (
    import_data,
    evaluate_model,
    calculate_perplexity,
)

# Ensure the working directory is set to the repo directory.
print(f"Working directory: {Path.cwd()}")

# ------------------------- #
# MLFlow experiment setup

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_run_id", help="The runid of the model for which training needs to be extended", required=True
)
parser.add_argument(
    "--run_description", help="A short description of the run", required=False
)
parser.add_argument("--start_epoch", help="start epoch number of the extended training", required=True)
# parser.add_argument("--end_epoch", help="end epoch number of the extended training", required=True)
args = parser.parse_args()

# import mlflow server config
dotenv.load_dotenv("RnD/Training_and_Evaluation/mlflow_pytorch/mlflow_config.env")
uri_host = os.getenv("HOST") + ":" + os.getenv("PORT")
print(f"Tracking server: {uri_host}")
# set tracking uri
mlflow.set_tracking_uri(uri=uri_host)
# client = mlflow.MlflowClient(tracking_uri=uri_host)
experiment = mlflow.get_experiment_by_name(name="GPT2-smalldataset-pytorch")
if experiment is None:
    raise ValueError("Incorrect experiment name specified")

# Validate run id
experiment_run_id = args.experiment_run_id if args.experiment_run_id else None
try:
    last_run = mlflow.get_run(experiment_run_id)

except Exception as err:
    raise err

# Validate epochs data
try:
    start_epoch = int(args.start_epoch)
    # end_epoch = int(args.end_epoch)
    # if end_epoch - start_epoch <= 0:
    #     raise ValueError("End epoch number must be greater than start epoch number")

    if 'end_epoch' in last_run.data.params:
        last_epoch_number = int(last_run.data.params['end_epoch'])
    else:
        last_epoch_number = int(last_run.data.params['epochs'])

    if start_epoch != last_epoch_number + 1:
        raise ValueError("Start epoch number must be equal to the last epoch number + 1")

except Exception as err:
    raise err

# Parse run description
# TODO
run_description = args.run_description

# Download model
model = mlflow.pytorch.load_model(f"runs:/{experiment_run_id}/models")
## set device
compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(compute_device)
# set to train mode
model.train()

# Download config
config_file_path = mlflow.artifacts.download_artifacts(f"runs:/{experiment_run_id}/llm_config.yaml")
with open(config_file_path, 'r') as file:
    llm_config =  yaml.safe_load(file.read())


# ------------------------- #
# import training parameters
with open("RnD/Training_and_Evaluation/training_parameters.yaml", "r") as file:
    train_params = yaml.safe_load(file.read())

    # add start and end epochs
    train_params['start_epoch'] = start_epoch
    end_epoch = start_epoch + train_params['epochs']
    train_params['end_epoch'] = end_epoch

    # update run description
    run_description = run_description + f" Start epoch: {start_epoch}, End epoch: {end_epoch}"


# ------------------------- #
# prepare data for training #
## import data
train_data = import_data(path="RnD/Dataloader/training_data/the-verdict.txt")

## set seed for reproducibility
torch.manual_seed(100)

## split into train and validation sets
train_data_size = len(train_data)
split_index = int(train_data_size * train_params["dataset"]["train_val_split_ratio"])
val_data = train_data[split_index:]
train_data = train_data[:split_index]

## create Tokenizer
tokenizer = Tokenizer.from_pretrained(train_params["tokenizer_model"])

## create torch Datasets
train_dataset = GPTDatasetV1(
    txt=train_data,
    tokenizer=tokenizer,
    max_length=llm_config["context_length"],
    stride=train_params["dataset"]["stride"],
    annotate="Train dataset",
)
val_dataset = GPTDatasetV1(
    txt=val_data,
    tokenizer=tokenizer,
    max_length=llm_config["context_length"],
    stride=train_params["dataset"]["stride"],
    annotate="Validation dataset",
)

del train_data, val_data  # to clear memory

## create torch Dataloaders
train_dataloader = create_dataloader_v1(
    dataset=train_dataset,
    batch_size=train_params["dataset"]["batch_size"],
    shuffle=True,
    drop_last=True,
    num_workers=0,
)
val_dataloader = create_dataloader_v1(
    dataset=val_dataset,
    batch_size=train_params["dataset"]["batch_size"],
    shuffle=False,
    drop_last=False,
    num_workers=0,
)

# ------------------------- #
# Training

## loss function
loss_func = torch.nn.CrossEntropyLoss()

## optimizer
learning_rate = train_params["optimizer"]["learning_rate"]
weight_decay = train_params["optimizer"]["weight_decay"]
optimizer = torch.optim.AdamW(
    params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
)

## metrics
metrics_func = Perplexity().to(device=compute_device)

with mlflow.start_run(
    experiment_id=experiment.experiment_id, description= run_description,
    nested=True, parent_run_id=experiment_run_id
):
    # log the LLM config
    mlflow.log_artifact(config_file_path)

    # log training parameters
    params_to_log = train_params.copy()
    params_to_log["loss_function"] = (loss_func.__class__.__name__,)
    params_to_log["optimizer"] = (optimizer.__class__.__name__,)
    mlflow.log_params(params=params_to_log)

    # log model summary
    with open(Path.cwd().joinpath("model_summary.txt"), "w") as f:
        f.write(str(summary(model=model)))
    mlflow.log_artifact(str(Path.cwd().joinpath("model_summary.txt")))

    epochs = train_params["epochs"]

    for epoch in range(start_epoch, end_epoch, 1):
        loss = 0
        for batch_num, (input_batch, target_batch) in enumerate(train_dataloader, 1):
            input_batch = input_batch.to(compute_device)
            target_batch = target_batch.to(compute_device)

            # zero the gradients
            optimizer.zero_grad()

            # model output
            logits = model(input_batch)

            # loss calculation
            loss_batch = loss_func(logits.flatten(0, 1), target_batch.flatten(0))

            # backpropagation
            loss_batch.backward()

            # weight update
            optimizer.step()

            loss += loss_batch.item()

        mlflow.log_metric("train_loss", value=loss / batch_num, step=epoch)
        mlflow.log_metric("train_perplexity", value=exp(loss / batch_num), step=epoch)

        # Validation metrics
        val_loss = evaluate_model(
            model=model, val_dataloader=val_dataloader, compute_device=compute_device
        )
        val_perplexity = calculate_perplexity(
            model=model,
            dataloader=val_dataloader,
            compute_device=compute_device,
            perpl_func=metrics_func,
        )
        mlflow.log_metric("validation_loss", value=val_loss, step=epoch)
        mlflow.log_metric("validation_perplexity", value=val_perplexity, step=epoch)

        print("\n---------------------------------------------")
        print(f"At the end of epoch: {epoch}")
        print(f"Training Loss: {loss / batch_num: .3f}")
        print(f"Training Perplexity: {exp(loss / batch_num): .3f}")
        print(f"Validation Loss: {val_loss: .3f}")
        print(f"Validation Perplexity: {val_perplexity: .3f}")

    # Save the trained model

    ## signature
    model.eval()
    for input_batch, target_batch in val_dataloader:
        with torch.no_grad():
            model_output = model(input_batch.to(compute_device))
            signature = mlflow.models.infer_signature(
                model_input=input_batch.numpy(),
                model_output=model_output.to("cpu").numpy(),
            )
        break

    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="models",
        registered_model_name="custom_gpt2_model",
        signature=signature,
        pip_requirements=["code/llmarch-0.0.1-py3-none-any.whl"],
        code_paths=["dist/llmarch-0.0.1-py3-none-any.whl"],
        # required when saving in eager mode.
        # Another option is to convert the model into torch.jit.trace
    )
