import torch
from RnD.LLM_arch.GPT2.llm_gpt2 import GPT2


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
    assert model.pos_embd_layer.weight.shape[0] == context_length, "Incorrect context length, please check the embedding size of the positional layer for confirmation"

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


if __name__ == "__main__":
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained("gpt2")

    test_txt = "Gudiya is "

    tokenized_text = tokenizer.encode(test_txt)
    tokenized_text_ids = torch.tensor(tokenized_text.ids).unsqueeze(dim=0)
    # reshaping to [batch_size, num_tokens]

    ##--------------##
    # GPT model
    import yaml

    with open("GPT2_arch_config.yaml", "r") as file:
        cfg = yaml.safe_load(file.read())

    gpt_model = GPT2(**cfg)
    gpt_model.eval()
    # Set to evaluation mode - required so that the dropout layers are disabled

    generated_text_token_ids = generate_tokens(
        model=gpt_model,
        input_tokens=tokenized_text_ids,
        context_length=cfg["context_length"],
        max_new_tokens=10,
    )
    print(f"Number of output tokens: {generated_text_token_ids.size()}")
    # convert ids to text
    decoded_text = tokenizer.decode(ids=generated_text_token_ids.squeeze(0).tolist())
    print(f"Decoded text: {decoded_text}")
