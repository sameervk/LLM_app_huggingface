"""
Script for pretraining a small GPT2 custom model on books from Project Gutenberg.
"""

import time
import os
from pathlib import Path
from argparse import ArgumentParser
import yaml
import torch
from tokenizers import Tokenizer


# import torchmetrics

from RnD.LLM_arch.GPT2.llm_gpt2 import GPT2
from RnD.Dataloader.dataloader import GPTDatasetV1, create_dataloader_v1
from RnD.Training_and_Evaluation.pytorch.utils import (
    calculate_loss_per_batch,
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
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    tokenizer: Tokenizer,
    llm_config: dict,
    train_params: dict,
    path_to_files: list,
    sample_prompt: str,
    model_version: str,
    model_save_dir: Path,
    debug=False,
    text_separator: str = "<|endoftext|>",
):
    total_files = len(path_to_files)
    train_loss_per_epoch, val_loss_per_epoch, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1  # for saving checkpoints primarily
    start_time = time.time()
    epoch = 0

    model.to(device)
    try:
        for epoch in range(1, num_epochs + 1):
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

                    # calculate train loss
                    train_loss = calculate_loss_per_batch(
                        x_batch=input_batch,
                        y_batch=target_batch,
                        model=model,
                        compute_device=device,
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
                        # a quick test to check if the function is working fine
                        return [train_loss.item()], [], [tokens_seen]

                # evaluate on validation data
                avg_val_loss_per_file = evaluate_model(
                    model=model, val_dataloader=val_dataloader, compute_device=device
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

            # save model
            file_name = (
                f"model_optim_v{model_version}_step-{global_step}_epoch-{epoch}.pth"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                },
                model_save_dir.joinpath(file_name),
            )
            print(f"Model {file_name} saved in {model_save_dir}")

            train_loss_per_epoch.append(
                sum(train_loss_per_batch) / len(train_loss_per_batch)
            )
            print(f"Training loss: {train_loss_per_epoch[epoch - 1]:.3f}")

            val_loss_per_epoch.append(sum(val_loss_per_file) / total_files)
            print(f"Validation loss: {val_loss_per_epoch[epoch - 1]:.3f}")

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

    except (Exception, KeyboardInterrupt) as err:
        print(err)
        file_name = f"model_optim_v{model_version}_step-{global_step}_epoch-{epoch}.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
            },
            model_save_dir.joinpath(file_name),
        )

    else:
        return train_loss_per_epoch, val_loss_per_epoch, track_tokens_seen


if __name__ == "__main__":
    argparser = ArgumentParser(description="Custom GPT2 model training configuration")

    argparser.add_argument(
        "--data_dir", type=str, help="path to the training data", required=True
    )

    argparser.add_argument(
        "--model_version", type=str, help="version of the model", required=True
    )
    argparser.add_argument(
        "--debug",
        type=bool,
        help="A quick test to check if model training code works",
        default=False,
        required=False,
    )
    args = argparser.parse_args()

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

    # -------------#
    # ----MODEL----#

    # Import LLM config
    with open("./llm_config.yaml", "r") as file:
        llm_config = yaml.safe_load(file.read())

    # -------------#
    # Initialise model
    model = GPT2(**llm_config)

    # get compute device
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get tokenizer
    tokenizer = Tokenizer.from_pretrained("gpt2")

    # Import training parameters
    with open("./training_parameters.yaml", "r") as file:
        training_params = yaml.safe_load(file.read())

    # Define Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_params["optimizer"]["learning_rate"],
        weight_decay=training_params["optimizer"]["weight_decay"],
    )

    # Directory to save
    model_save_dir = Path(os.path.join(os.curdir, "./saved_models"))
    if not model_save_dir.exists():
        os.mkdir(model_save_dir)

    model_version = args.model_version
    model_ver_save_dir = model_save_dir.joinpath(model_version)
    if not model_ver_save_dir.exists():
        os.mkdir(model_ver_save_dir)

    training_losses, validation_losses, tokens_seen_with_epoch = train_model_basic(
        model=model,
        optimizer=optimizer,
        device=compute_device,
        num_epochs=training_params["epochs"],
        tokenizer=tokenizer,
        debug=args.debug,
        text_separator="<|endoftext|>",
        llm_config=llm_config,
        train_params=training_params,
        sample_prompt="Every day ",
        path_to_files=all_files,
        model_version=model_version,
        model_save_dir=model_ver_save_dir,
    )

# TODO
# 2. Need to check how to save artifacts such as llm and training config without using mlflow
