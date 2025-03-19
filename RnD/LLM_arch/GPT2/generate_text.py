import torch
from typing import Dict
from RnD.LLM_arch.GPT2.llm_gpt2 import GPT2

def generate_text(model: torch.nn.Module,
                  input_tokens: torch.Tensor,
                  context_length: int,
                  max_new_tokens: int
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


