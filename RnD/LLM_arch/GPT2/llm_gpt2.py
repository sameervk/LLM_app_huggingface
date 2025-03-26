import torch

from RnD.LLM_arch.GPT2.transformer_block_gpt2 import TransformerBlockGPT2


class GPT2(torch.nn.Module):
    def __init__(self, **cfg):
        super().__init__()

        self.num_transform_blocks = cfg["num_layers"]

        self.token_embd_layer = torch.nn.Embedding(
            num_embeddings=cfg["vocab_size"], embedding_dim=cfg["embed_dim"]
        )
        self.pos_embd_layer = torch.nn.Embedding(
            num_embeddings=cfg["context_length"], embedding_dim=cfg["embed_dim"]
        )
        self.dropout_layer = torch.nn.Dropout(p=cfg["drop_rate"])

        self.transformer_blocks = torch.nn.Sequential(
            *[TransformerBlockGPT2(**cfg) for _ in range(self.num_transform_blocks)]
        )

        self.layer_norm = torch.nn.LayerNorm(normalized_shape=cfg["embed_dim"])

        self.final_linear_layer = torch.nn.Linear(
            in_features=cfg["embed_dim"], out_features=cfg["vocab_size"], bias=False
        )

    def forward(self, X_tokens):
        batch_size, input_token_size = X_tokens.shape

        token_embeds = self.token_embd_layer(X_tokens)
        pos_embeds = self.pos_embd_layer(
            torch.arange(input_token_size, device=X_tokens.device)
        )

        X = token_embeds + pos_embeds
        X = self.dropout_layer(X)
        X = self.transformer_blocks(X)
        X = self.layer_norm(X)
        logits = self.final_linear_layer(X)

        return logits


if __name__ == "__main__":
    import yaml

    # from tokenizers import Tokenizer
    #
    # gpt_tokenizer = Tokenizer.from_pretrained("gpt2")
    #
    # test_text = ["How are you doing?",
    #              "What is up?"
    #             ]
    #
    # test_tokens = gpt_tokenizer.encode_batch(test_text)
    # print(test_tokens)

    with open("GPT2_arch_config.yaml", "r") as file:
        cfg = yaml.safe_load(file.read())

    batch_size = 2
    num_tokens = 4
    test_input_tokens = torch.randint(
        0, cfg["vocab_size"], size=(batch_size, num_tokens)
    )
    print(f"Input shape: {test_input_tokens.shape}")

    gpt_model = GPT2(**cfg)

    output = gpt_model(test_input_tokens)
    print(f"Output shape: f{output.shape}")

    ##---------------------##
    # Number of parameters
    ##---------------------##
    num_param_list = []
    [num_param_list.append(p.numel()) for p in gpt_model.parameters()]
    total_parameters = sum(num_param_list)
    print(f"Total number of parameters: {total_parameters}")

    num_params_feedforward = 0
    num_params_attention = 0
    for name, param in gpt_model.named_parameters():
        if "feedforward" in name:
            num_params_feedforward += param.numel()
        elif "multihead" in name:
            num_params_attention += param.numel()
    print(f"Number of parameters in feed forward layers: {num_params_feedforward}")
    print(f"Number of parameters in attention layers: {num_params_attention}")

    ##---------------------##
    # Model size
    ##---------------------##

    total_size_bytes = total_parameters * 4
    # assuming float32
    print(f"Total memory size: {total_size_bytes / (1024 * 1024):.2f} MB")
