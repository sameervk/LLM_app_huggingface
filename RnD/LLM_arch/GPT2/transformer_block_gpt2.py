import torch
from RnD.Attention_Mechanism.multihead_attention import MultiHeadAttention
from RnD.LLM_arch.GPT2.feed_forward_layer import FeedForward


class TransformerBlockGPT2(torch.nn.Module):
    def __init__(self, **cfg):
        super().__init__()

        assert cfg["embed_dim"] % cfg["num_attention_heads"] == 0, (
            "Embedding dimension must be divisible by number of heads for "
            "this architecture"
        )

        self.multihead_attention_layer = MultiHeadAttention(
            embed_dim=cfg["embed_dim"],
            head_dim_out=int(cfg["embed_dim"] / cfg["num_attention_heads"]),
            num_heads=cfg["num_attention_heads"],
            context_len=cfg["context_length"],
            dropout_p=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.feedforward_layer = FeedForward(input_dim=cfg["embed_dim"])

        self.dropout_layer_1 = torch.nn.Dropout(p=cfg["drop_rate"])
        self.layer_norm_1 = torch.nn.LayerNorm(cfg["embed_dim"])

        self.dropout_layer_2 = torch.nn.Dropout(p=cfg["drop_rate"])
        self.layer_norm_2 = torch.nn.LayerNorm(cfg["embed_dim"])

    def forward(self, X):
        skip_connection = X
        X = self.layer_norm_1(X)
        X = self.multihead_attention_layer(X)
        X = self.dropout_layer_1(X)
        X = X + skip_connection

        skip_connection = X
        X = self.layer_norm_2(X)
        X = self.feedforward_layer(X)
        X = self.dropout_layer_2(X)

        X = X + skip_connection

        return X


if __name__ == "__main__":
    import yaml

    with open("GPT2_arch_config.yaml", mode="r") as file:
        cfg = yaml.safe_load(file.read())
    print(cfg)

    # test input
    batch_size = 3
    num_tokens = 5
    test_input = torch.rand(size=(batch_size, num_tokens, cfg["embed_dim"]))

    # transformer layer
    test_transformer_layer = TransformerBlockGPT2(**cfg)

    test_output = test_transformer_layer(test_input)

    print(f"Test output shape: f{test_output.size()}")
