from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from torch.nn import Module
from torch import device

from RnD.LLM_arch.GPT2.generate_text import generate_tokens

def calculate_loss(model:Module,
                   dataloader: DataLoader,
                   num_batches: int,
                   compute_device: device
                   ):

    num_batches = min(num_batches, len(dataloader))
    assert num_batches>0, "Number of batches must be greater than zero"

    # loss per epoch
    avg_loss = 0.0
    for batch, (x_batch, y_batch) in enumerate(dataloader):

        if batch<num_batches:

            loss_per_batch = calculate_loss_per_batch(x_batch=x_batch,
                                                      y_batch=y_batch,
                                                      model=model,
                                                      compute_device=compute_device
                                                      )
            avg_loss += loss_per_batch.item()

        else:
            break

    return avg_loss / num_batches


def calculate_loss_per_batch(x_batch,
                             y_batch,
                             model,
                             compute_device
                             ):

    x_batch = x_batch.to(compute_device) # transfer to the device
    # shape:[batch_size, num_tokens]
    y_batch = y_batch.to(compute_device)
    # shape: [batch_size, num_tokens]

    logits = model(x_batch)
    # shape:[batch_size, context_length, embed_dim]

    loss_per_batch = cross_entropy(
        input=logits.flatten(0, 1),
        target=y_batch.flatten(0)
    )

    return loss_per_batch


def generate_text(model: Module,
                  prompt:str,
                  num_output_tokens:int
                  ):

    model.eval()

