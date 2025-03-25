import yaml
from tokenizers import Tokenizer
import torch
import mlflow

from RnD.Tokenisation_Embedding.dataloader import GPTDatasetV1, create_dataloader_v1
from RnD.LLM_arch.GPT2.llm_gpt2 import GPT2
from RnD.Training_and_Evaluation.utils import (
    import_data,
    generate_text,
    evaluate_model,
    calculate_loss_per_batch,
)


if __name__=="__main__":

    import time
    start_time = time.time()

    # identify training hardware: cpu or gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # for reproducibility
    torch.manual_seed(100)

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
    print(f"Training will be executed on {device}")
    model.to(device=device)

    # loss before training
    train_loss = evaluate_model(model=model,
                                          val_dataloader=train_dataloader,
                                          compute_device=device
                                          )
    val_loss = evaluate_model(
        model=model,
        val_dataloader=val_dataloader,
        compute_device=device,
    )
    print(f"\nTrain data loss before training: {train_loss:.4f}")
    print(f"Validation data loss before training: {val_loss:.4f}")

    # Evaluation using a prompt
    test_prompt = "What is "
    generated_text = generate_text(model=model,
                                   prompt=test_prompt,
                                   num_output_tokens=20,
                                   tokenizer=tokenizer,
                                   context_len=llm_config["context_length"],
                                   compute_device=device
                                   )
    print(f"\nGenerated text: {generated_text}")

    # optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=train_params["optimizer"]["learning_rate"],
                                  weight_decay=train_params["optimizer"]["weight_decay"]
                                  )

    epochs = train_params["epochs"]

    # frequency at which model evaluation is carried out during training
    eval_freq = 5
    model.train()
    loss_at_epoch = []
    for epoch in range(epochs):

        epoch_loss = 0

        for batch_num, (input_batch, target_batch) in enumerate(train_dataloader, 1):

            # reset the gradients to zero
            optimizer.zero_grad()

            # calculate loss per batch
            loss = calculate_loss_per_batch(x_batch=input_batch,
                                            y_batch=target_batch,
                                            model=model,
                                            compute_device=device
                                            )
            epoch_loss += loss.item()

            # calculate the gradients and propagate the loss backwards
            loss.backward()

            # update the weights
            optimizer.step()

            # show validation and
            if batch_num % eval_freq == 0:
                eval_loss = evaluate_model(model=model,
                                           val_dataloader=val_dataloader,
                                           compute_device=device
                                           )
                print(f"Validation data loss at epoch {epoch}: {eval_loss:.3f}")

        loss_at_epoch.append(epoch_loss/batch_num)
        print(f"\nTraining loss at epoch {epochs}: {epoch_loss/batch_num:.3f}")


    end_time = time.time()
    print(f"\nTime taken to run: {end_time-start_time} seconds.")
