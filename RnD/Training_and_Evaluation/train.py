import yaml
from pathlib import Path
from tokenizers import Tokenizer
import torch
import mlflow

from RnD.Tokenisation_Embedding.dataloader import GPTDatasetV1, create_dataloader_v1
from RnD.LLM_arch.GPT2.llm_gpt2 import GPT2
from RnD.Training_and_Evaluation.utils import calculate_loss
from RnD.LLM_arch.GPT2.generate_text import generate_text


# import data
def import_data(path: str):
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
            return txt
    else:
        raise FileNotFoundError(path)



if __name__=="__main__":

    import time
    start_time = time.time()

    # import tokenizer
    tokenizer_model = "gpt2"
    tokenizer = Tokenizer.from_pretrained(tokenizer_model)

    # import model config
    with open("RnD/LLM_arch/GPT2/GPT2_arch_config.yaml", "r") as f:
        llm_config = yaml.safe_load(f.read())

    # import training parameters config file
    with open("RnD/Training_and_Evaluation/training_parameters.yaml", "r") as f:
        train_params = yaml.safe_load(f.read())

    # import data
    # set the root directory as the working directory
    training_data_path = "RnD/Tokenisation_Embedding/the-verdict.txt"
    training_data = import_data(path=training_data_path)
    print(f"Number of characters: {len(training_data)}")

    # split into train and validation
    train_validation_split_ratio = train_params["dataset"]["train_val_split_ratio"]
    train_data_length = int(train_validation_split_ratio*len(training_data))
    val_data = training_data[train_data_length:]
    training_data = training_data[: train_data_length]

    # for reproducibility
    torch.manual_seed(100)

    # create Pytorch Datasets
    train_dataset = GPTDatasetV1(txt=training_data,
                                 tokenizer=tokenizer,
                                 max_length=llm_config["context_length"],
                                 stride=train_params["dataset"]["stride"]
                                 )
    val_dataset = GPTDatasetV1(txt=val_data,
                               tokenizer=tokenizer,
                               max_length=llm_config["context_length"],
                               stride=train_params["dataset"]["stride"]
                               )

    # create Pytorch Dataloaders for training and evaluation
    batch_size = train_params["dataset"]["batch_size"]
    train_dataloader = create_dataloader_v1(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=0
                                            )
    val_dataloader = create_dataloader_v1(dataset=val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=0
                                          )
    ## checking
    for x, y in train_dataloader:
        print(f"\nTraining dataloader - X: {x.shape}, Y:{y.shape}")
        print(f"X: {x[0][:10]}, Y: {y[0][:10]}")
        break
    for x, y in val_dataloader:
        print(f"\nValidation dataloader - X: {x.shape}, Y:{y.shape}")
        break

    # initiate model
    model = GPT2(**llm_config)

    # identify training hardware: cpu or gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will be executed on {device}")
    model.to(device=device)

    # loss before training
    with torch.no_grad():
        train_loss = calculate_loss(model=model,
                                  dataloader=train_dataloader,
                                  num_batches=len(train_dataloader),
                                  compute_device=device
                                  )
        print(f"Train data loss before training: {train_loss:.4f}")

        val_loss = calculate_loss(model=model,
                                  dataloader=val_dataloader,
                                  num_batches=len(val_dataloader),
                                  compute_device=device
                                  )
        print(f"Validation data loss before training: {val_loss:.4f}")

    # code for evaluation
    model.eval()
    test_text = "What is "
    test_text_ids = torch.tensor(tokenizer.encode(test_text).ids).unsqueeze(0).to(device)
    with torch.no_grad():
        generated_tokens = generate_text(model=model,
                                         input_tokens=test_text_ids,
                                         context_length=llm_config["context_length"],
                                         max_new_tokens=20
                                         )
    generated_text = tokenizer.decode(generated_tokens.squeeze(0).tolist())
    print(f"{generated_text}")


    end_time = time.time()
    print(f"\nTime taken to run: {end_time-start_time} seconds.")
