"""
Script for pretraining a small GPT2 custom model on books from Project Gutenberg.
"""

import time
import os
import dotenv
from pathlib import Path
from argparse import ArgumentParser
import yaml
import torch
from tokenizers import Tokenizer
import mlflow
from torchinfo import summary
from math import exp

from RnD.LLM_arch.GPT2.llm_gpt2 import GPT2
from RnD.Dataloader.dataloader import GPTDatasetV1, create_dataloader_v1
from RnD.Training_and_Evaluation.pytorch.utils import (
    evaluate_model,
    generate_text,
)
from RnD.Training_and_Evaluation.project_gutenberg.utils import print_eta


def read_text_tile(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    return text_data


def create_dataloaders(
    text_data, tokenizer, train_ratio, batch_size, max_length, stride, num_workers=0
):
    split_index = int(len(text_data) * train_ratio)

    # create Pytorch Datasets
    train_dataset = GPTDatasetV1(
        txt=text_data[:split_index],
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        annotate="training dataset",
    )
    val_dataset = GPTDatasetV1(
        txt=text_data[split_index:],
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        annotate="validation dataset",
    )

    train_dataloader = create_dataloader_v1(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    val_dataloader = create_dataloader_v1(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader


def train_model_basic(
    mlflow_exp_id: str,
    mlflow_run_name: str | None,
    nested_run: bool,
    parent_run_id: str | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tokenizer: Tokenizer,
    llm_config: dict,
    train_params: dict,
    path_to_files: list,
    sample_prompt: str,
    debug=False,
    text_separator: str = "<|endoftext|>",
    model_save_dir: Path = Path("."),
):
    total_files = len(path_to_files)
    train_loss_per_epoch, val_loss_per_epoch, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1  # for saving checkpoints primarily
    start_time = time.time()

    start_epoch = train_params["start_epoch"]
    num_epochs = train_params["epochs"]

    with mlflow.start_run(
        experiment_id=mlflow_exp_id,
        run_name=mlflow_run_name,
        nested=nested_run,
        parent_run_id=parent_run_id,
    ):
        try:
            # CHECKPOINTS SAVE DIRECTORY
            current_run = mlflow.active_run()
            if not nested_run:
                model_checkpoint_dir = model_save_dir.joinpath(current_run.info.run_id)
                if not model_checkpoint_dir.exists():
                    os.mkdir(model_checkpoint_dir)
            else:
                model_checkpoint_dir = model_save_dir.joinpath(parent_run_id).joinpath(
                    current_run.info.run_id
                )
                if not model_checkpoint_dir.exists():
                    os.mkdir(model_checkpoint_dir)

            # log training parameters
            mlflow.log_params(train_params)

            # log model summary
            with open(model_checkpoint_dir.joinpath("model_summary.txt"), "w") as f:
                f.write(str(summary(model=model)))
            mlflow.log_artifact(str(model_checkpoint_dir.joinpath("model_summary.txt")))

            model.to(device)

            loss_func = torch.nn.CrossEntropyLoss()

            for epoch in range(start_epoch, start_epoch + num_epochs):
                print("\n\n-----------------------------")
                print(f"Training epoch: {epoch}")

                train_loss_per_batch = []
                val_loss_per_file = []

                # Iterate over the books in the training corpus
                for index, file_path in enumerate(path_to_files, 1):
                    book_start_time = time.time()
                    text_data = read_text_tile(file_path) + f" {text_separator} "
                    print(f"Tokenizing file {index} of {total_files}: {file_path}")

                    # Initiate data loaders
                    train_dataloader, val_dataloader = create_dataloaders(
                        text_data=text_data,
                        tokenizer=tokenizer,
                        train_ratio=train_params["dataset"]["train_val_split_ratio"],
                        batch_size=train_params["dataset"]["batch_size"],
                        max_length=llm_config["context_length"],
                        stride=train_params["dataset"]["stride"],
                        num_workers=0,
                    )
                    print("\nTraining...")
                    model.train()

                    for batch_num, (input_batch, target_batch) in enumerate(
                        train_dataloader, 1
                    ):
                        # reset the gradients to zero
                        optimizer.zero_grad()

                        input_batch = input_batch.to(device)
                        target_batch = target_batch.to(device)

                        logits = model(input_batch)

                        # calculate train loss
                        train_loss = loss_func(
                            logits.flatten(0, 1), target_batch.flatten(0)
                        )

                        train_loss_per_batch.append(train_loss.item())

                        # backpropagate the gradients
                        train_loss.backward()

                        # update the weights
                        optimizer.step()

                        # count the number of tokens seen in the batch
                        tokens_seen += input_batch.numel()

                        # update the global step num
                        global_step += 1

                        if debug:
                            print("In Debug mode")
                            # a quick test to check if the function is working fine
                            return [train_loss.item()], [], [tokens_seen]

                    # evaluate on validation data
                    avg_val_loss_per_file = evaluate_model(
                        model=model,
                        val_dataloader=val_dataloader,
                        compute_device=device,
                    )
                    val_loss_per_file.append(avg_val_loss_per_file)

                    # log the remaining books left
                    print_eta(
                        start_time=start_time,
                        book_start_time=book_start_time,
                        index=index,
                        total_files=total_files,
                    )

                print(f"\n\nAt the end of epoch: {epoch}")

                train_loss_per_epoch.append(
                    sum(train_loss_per_batch) / len(train_loss_per_batch)
                )

                mlflow.log_metric(
                    "train_loss", value=train_loss_per_epoch[-1], step=epoch
                )
                train_perplexity = int(exp(train_loss_per_epoch[-1]))
                mlflow.log_metric(
                    "train_perplexity", value=train_perplexity, step=epoch
                )
                print(
                    f"Training loss: {train_loss_per_epoch[-1]:.3f}, Perplexity: {train_perplexity}"
                )

                val_loss_per_epoch.append(sum(val_loss_per_file) / total_files)
                mlflow.log_metric("val_loss", value=val_loss_per_epoch[-1], step=epoch)
                val_perplexity = int(exp(val_loss_per_epoch[-1]))
                mlflow.log_metric("val_perplexity", value=val_perplexity, step=epoch)
                print(
                    f"Validation loss: {val_loss_per_epoch[-1]:.3f}, Perplexity: {val_perplexity}"
                )

                track_tokens_seen.append(tokens_seen)  # this is cumulative per epoch
                print(f"Tokens seen: {tokens_seen}\n")

                # Generate sample output given a context
                sample_generated_text = generate_text(
                    model=model,
                    prompt=sample_prompt,
                    num_output_tokens=50,
                    tokenizer=tokenizer,
                    context_len=llm_config["context_length"],
                    compute_device=device,
                )
                print(
                    f"\nGenerated text after epoch {epoch} given prompt '{sample_prompt}' : {sample_generated_text}"
                )

                # save model
                file_name = f"model_optim_epoch-{epoch}_step-{global_step}.pth"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                    },
                    model_checkpoint_dir.joinpath(file_name),
                )
                print(f"Model {file_name} saved in {model_checkpoint_dir}")

        except (Exception, KeyboardInterrupt) as err:
            print(f"\nError: {err}")
            file_name = f"model_optim_epoch-{epoch}_step-{global_step}.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                },
                model_checkpoint_dir.joinpath(file_name),
            )
            mlflow.log_artifact(str(model_checkpoint_dir.joinpath(file_name)))

            raise err
            # return train_loss_per_epoch, val_loss_per_epoch, track_tokens_seen

        else:
            model.eval()
            with torch.no_grad():
                model_output = model(input_batch.to(compute_device))
                model_signature = mlflow.models.infer_signature(
                    model_input=input_batch.to("cpu").numpy(),
                    model_output=model_output.to("cpu").numpy(),
                )
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="models",
                registered_model_name="custom_gpt2_model",
                signature=model_signature,
                pip_requirements=["code/llmarch-0.0.1-py3-none-any.whl"],
                code_paths=["dist/llmarch-0.0.1-py3-none-any.whl"],
            )

            return train_loss_per_epoch, val_loss_per_epoch, track_tokens_seen


if __name__ == "__main__":
    argparser = ArgumentParser(description="Custom GPT2 model training configuration")

    argparser.add_argument(
        "--experiment_name", type=str, help="mlflow experiment name", required=True
    )
    argparser.add_argument(
        "--run_name", type=str, help="mlflow experiment run name", required=False
    )
    argparser.add_argument(
        "--nested_run_id", type=str, help="Parent run id of nested run", required=False
    )
    argparser.add_argument(
        "--load_checkpoint_path",
        type=str,
        help="Path to torch checkpoint .pth to load into the model",
        required=False,
    )

    argparser.add_argument(
        "--data_dir", type=str, help="path to the training data", required=True
    )

    # add '--debug` as parameter to the script for debugging
    argparser.add_argument(
        "--debug",
        help="A quick test to check if model training code works",
        action="store_true",
    )
    args = argparser.parse_args()

    # first check if the working directory is set correctly
    # the working directory is the root repo containing the mlflow directories: mlruns, mlarfifacts, etc.
    if not str(Path.cwd()).endswith("LLM_app_huggingface"):
        raise Exception("Set the working directory to the root directory")
    training_dir = Path.cwd().joinpath("RnD/Training_and_Evaluation/project_gutenberg")
    if not training_dir.exists():
        raise NotADirectoryError("Check the directory containing the code")

    # --------------------------#
    # ----PATH TO TEXT FILES----#
    # Get the list of paths of all the data files
    # data_dir = "../../Dataloader/training_data"
    data_dir = args.data_dir

    if not Path(data_dir).exists():
        raise NotADirectoryError("Data directory does not exist.")
    all_files = [
        os.path.join(path, name)
        for path, subdirs, files in os.walk(data_dir)
        for name in files
        if name.endswith((".txt"))
    ]

    # --------------------#
    # MLFLOW CONFIG SETUP
    env_var = dotenv.load_dotenv(training_dir.joinpath(".env"))
    if not env_var:
        raise FileNotFoundError("Environment variables file .env not found")

    # tracking uri
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)

    # experiment name - remove any whitespaces
    experiment_name = args.experiment_name.replace(" ", "")
    current_exp = mlflow.get_experiment_by_name(experiment_name)
    if current_exp is None:
        mlflow.set_experiment(experiment_name=experiment_name)
        # set experiment
        experiment_tags = {
            "mlflow.note.content": "Training of GPT2 architecture-based LLM using Pytorch. "
            "Data: Project Gutenberg: https://github.com/pgcorpus/gutenberg"
        }
        mlflow.set_experiment_tags(tags=experiment_tags)
        current_exp = mlflow.get_experiment_by_name(experiment_name)
    # run name
    exp_run_name = args.run_name.replace(" ", "")
    if exp_run_name == "":
        exp_run_name = None

    # if nested
    parent_run_id = args.nested_run_id if args.nested_run_id else None
    if parent_run_id is not None:
        try:
            mlflow.get_run(parent_run_id)
        except Exception:
            raise Exception(f"Parent run id: {parent_run_id} does not exist")
        else:
            nested = True
    else:
        nested = False

    # -------------#
    # ----MODEL----#

    # Import LLM config
    with open(training_dir.joinpath("llm_config.yaml"), "r") as file:
        llm_config = yaml.safe_load(file.read())

    # Import training parameters
    with open(training_dir.joinpath("training_parameters.yaml"), "r") as file:
        training_params = yaml.safe_load(file.read())

    # get compute device
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if checkpoint to be loaded
    checkpoint_path = args.load_checkpoint_path if args.load_checkpoint_path else ""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=compute_device)
        load_checkpoint = True
    else:
        load_checkpoint = False

    # Initialise model
    model = GPT2(**llm_config)

    if load_checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(compute_device)

    # Define Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_params["optimizer"]["learning_rate"],
        weight_decay=training_params["optimizer"]["weight_decay"],
    )
    if load_checkpoint:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])

    # This does not work - throws error stating that optimizer is not on the right device.
    # It appears that first the checkpoint must be loaded, then the model and then the optimizer must be defined before
    # loading the state into it as in the above code
    # # if checkpoint to be loaded
    # checkpoint_path = args.load_checkpoint_path if args.load_checkpoint_path else ""
    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path, map_location="cpu")
    #     model.load_state_dict(checkpoint["model_state_dict"])
    #     model.to(compute_device)
    #     optimizer.load_state_dict(checkpoint["optim_state_dict"])

    # get tokenizer
    tokenizer = Tokenizer.from_pretrained("gpt2")

    # Directory to save
    model_save_dir = training_dir.joinpath("saved_models")
    if not model_save_dir.exists():
        os.mkdir(model_save_dir)

    # Training and saving
    training_losses, validation_losses, tokens_seen_with_epoch = train_model_basic(
        mlflow_exp_id=current_exp.experiment_id,
        mlflow_run_name=exp_run_name,
        nested_run=nested,
        parent_run_id=parent_run_id,
        model=model,
        optimizer=optimizer,
        device=compute_device,
        tokenizer=tokenizer,
        llm_config=llm_config,
        train_params=training_params,
        path_to_files=all_files,
        sample_prompt="Every day ",
        debug=args.debug,
        text_separator="<|endoftext|>",
        model_save_dir=model_save_dir,
    )
