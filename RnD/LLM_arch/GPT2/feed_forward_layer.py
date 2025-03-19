import torch


class FeedForward(torch.nn.Module):

    def __init__(self, input_dim:int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = 4 * self.input_dim

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.output_dim), # expand
            torch.nn.GELU(),
            torch.nn.Linear(self.output_dim, self.input_dim)
        )

    def forward(self, X):

        return self.layers(X)


if __name__=="__main__":

    batch_size=2
    token_len = 5
    embed_size = 8

    test_cfg ={"input_dim": embed_size}

    test_layer = FeedForward(**test_cfg)

    torch.manual_seed(100)
    test_input = torch.rand(size=(batch_size, token_len, embed_size))

    output = test_layer(test_input)

    print(f"FeedForward output dim: {output.size()}")
