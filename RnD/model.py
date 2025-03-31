import mlflow
import os

import torch.cuda
from tokenizers import Tokenizer


def generate_tokens(
    model: torch.nn.Module,
    input_tokens: torch.Tensor,
    context_length: int,
    max_new_tokens: int,
):
    """

    :param model: LLM model
    :param input_tokens: shape - [batch_size, num_tokens]
    :param context_length: integer
    :param max_new_tokens: integer
    :return:
        [batch_size, max_new_tokens]
    """

    for _ in range(max_new_tokens):
        # cut the input size to context length
        input_tokens_cut = input_tokens[:, -context_length:]

        with torch.no_grad():
            logits = model(input_tokens_cut)
            # shape: [batch_size, num_tokens, vocab_size]

        # take the last row from the output sequence
        logits = logits[:, -1, :]
        # shape: [batch_size, 1, vocab_size]

        # apply softmax
        probabilities = torch.softmax(logits, dim=-1)
        # shape: [batch_size, 1, vocab_size]

        # take the index of the elem with the highest probability
        generated_idx = torch.argmax(probabilities, dim=-1, keepdim=True)
        # shape: [batch_size, 1]

        # add this to the input sequence
        input_tokens = torch.cat((input_tokens, generated_idx), dim=-1)

    return input_tokens

def generate_text(
    model: torch.nn.Module,
    prompt: str,
    num_output_tokens: int,
    tokenizer: Tokenizer,
    context_len: int,
    compute_device: torch.device,
) -> str:
    # encode to ids
    input_tokens = tokenizer.encode(prompt).ids

    # convert to tensor, add a batch dim, send to compute device
    input_tokens = torch.tensor(input_tokens).unsqueeze(0).to(compute_device)

    # put the model in evaluation mode to disable dropout
    model.eval()

    # generate the output tokens given input tokens
    generated_tokens = generate_tokens(
        model=model,
        input_tokens=input_tokens,
        context_length=context_len,
        max_new_tokens=num_output_tokens,
    )

    # decode to text
    generated_text = tokenizer.decode(generated_tokens.squeeze(0).tolist())

    # switch the model back to train mode
    model.train()

    return generated_text

if __name__=="__main__":

    # set tracking uri
    # mlflow.set_tracking_uri(uri="http://localhost/8080")
    model_uri = "mlartifacts/0/f899d06791074c7c9d5a9303c93deaad/artifacts/models"
    # model_uri = "models:/custom_gpt2_model@production"

    loaded_model = mlflow.pytorch.load_model(model_uri)

    test_prompt = "What is "
    tokenizer = Tokenizer.from_pretrained("gpt2")
    compute_device = torch.device("cpu")

    loaded_model.to(compute_device)

    generated_text = generate_text(
        model=loaded_model,
        prompt=test_prompt,
        num_output_tokens=20,
        tokenizer=tokenizer,
        context_len=256,
        compute_device=compute_device,
    )
    print(
        f"\nGenerated text before training given prompt '{test_prompt}' : {generated_text}"
    )