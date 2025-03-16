import torch

class SelfAttentionV1(torch.nn.Module):
    """
    uses Parameter class to initialise the Q, K and V weight matrices
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()

        self.W_query = torch.nn.Parameter(torch.rand(size=(dim_in, dim_out)))
        self.W_key = torch.nn.Parameter(torch.rand(size=(dim_in, dim_out)))
        self.W_value = torch.nn.Parameter(torch.rand(size=(dim_in, dim_out)))

    def forward(self, x_inputs):
        """
        A class of type torch.nn.Module must have a forward method
        :param x_inputs: input embeddings; shape: context_length x dim_in
        :return: z context vector; shape: context_length x dim_out
        """

        keys = x_inputs @ self.W_key # shape: context_length x dim_out
        queries = x_inputs @ self.W_query # shape: context_length x dim_out
        values =  x_inputs @ self.W_value # shape: context_length x dim_out

        attention_scores = queries @ keys.T # shape: context_length x context_length

        attention_weights = torch.softmax(attention_scores/keys.shape[-1]**0.5, dim=-1)
        # shape: context_length x context_length

        context_vec = attention_weights @ values
        # shape: context_length x dim_out

        return context_vec


class SelfAttentionV2(torch.nn.Module):
    """
    Uses Linear layers to initialise the Q, K and V weight matrices.
    More stable and effective for model training
    """

    def __init__(self, dim_in: int, dim_out: int, qkv_bias: bool):
        super().__init__()

        self.W_query = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)

    def forward(self, x_inputs):
        """
        A class of type torch.nn.Module must have a forward method
        :param x_inputs: input embeddings; shape: context_length x dim_in
        :return: z context vector; shape: context_length x dim_out
        """

        keys = self.W_key(x_inputs) # shape: context_length x dim_out
        queries = self.W_query(x_inputs) # shape: context_length x dim_out
        values = self.W_value(x_inputs) # shape: context_length x dim_out

        attention_scores = queries @ keys.T # shape: context_length x context_length

        attention_weights = torch.softmax(attention_scores/keys.shape[-1]**0.5, dim=-1)
        # shape: context_length x context_length

        context_vec = attention_weights @ values
        # shape: context_length x dim_out

        return context_vec


if __name__=="__main__":

    torch.manual_seed(123)

    # context_length x embedding_dim
    embedding_dim = 4
    test_inputs = torch.rand(size=(8, embedding_dim))

    self_attention_layer = SelfAttentionV2(dim_in=embedding_dim, dim_out=3, qkv_bias=False)

    test_output = self_attention_layer(test_inputs)

    print(test_output.shape)
