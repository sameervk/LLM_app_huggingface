from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from tokenizers import Tokenizer

from RnD.LLM_arch.GPT2.generate_text import generate_tokens


# import data
def import_data(path: str):
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
            return txt
    else:
        raise FileNotFoundError(path)


def calculate_loss(model:torch.nn.Module,
                   dataloader: DataLoader,
                   compute_device: torch.device
                   ) -> float:
    assert len(dataloader) >= 1, "Batch size must at least be 1"
    # loss per epoch
    avg_loss = 0.0
    for batch, (x_batch, y_batch) in enumerate(dataloader, 1):


        loss_per_batch = calculate_loss_per_batch(x_batch=x_batch,
                                                  y_batch=y_batch,
                                                  model=model,
                                                  compute_device=compute_device
                                                  )
        avg_loss += loss_per_batch.item()

    return avg_loss / batch


def calculate_loss_per_batch(x_batch,
                             y_batch,
                             model,
                             compute_device
                             ) -> torch.Tensor:

    x_batch = x_batch.to(compute_device) # transfer to the device
    # shape:[batch_size, num_tokens]
    y_batch = y_batch.to(compute_device)
    # shape: [batch_size, num_tokens]

    logits = model(x_batch)
    # shape:[batch_size, context_length/num_tokens, embed_dim]

    loss_per_batch = cross_entropy(
        input=logits.flatten(0, 1),
        target=y_batch.flatten(0)
    )

    return loss_per_batch


def evaluate_model(model:torch.nn.Module,
                   val_dataloader: DataLoader,
                   compute_device:torch.device
                   ) -> float:
    model.eval()
    with torch.no_grad():

        val_loss = calculate_loss(model=model,
                                  dataloader=val_dataloader,
                                  compute_device=compute_device
                                 )
    # set the model back to train mode
    model.train()

    return val_loss



def generate_text(model: torch.nn.Module,
                  prompt:str,
                  num_output_tokens:int,
                  tokenizer: Tokenizer,
                  context_len: int,
                  compute_device: torch.device
                  ) -> str:

    # encode to ids
    input_tokens = tokenizer.encode(prompt).ids

    # convert to tensor, add a batch dim, send to compute device
    input_tokens = torch.tensor(input_tokens).unsqueeze(0).to(compute_device)

    # put the model in evaluation mode to disable dropout
    model.eval()

    # generate the output tokens given input tokens
    generated_tokens = generate_tokens(model=model,
                                       input_tokens=input_tokens,
                                       context_length=context_len,
                                       max_new_tokens=num_output_tokens
                                       )

    # decode to text
    generated_text = tokenizer.decode(generated_tokens.squeeze(0).tolist())

    # switch the model back to train mode
    model.train()

    return generated_text


