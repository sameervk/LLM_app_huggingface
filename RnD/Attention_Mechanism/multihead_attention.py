import torch
from RnD.Attention_Mechanism.masked_attention import MaskedAttention


class MultiHeadAttentionWrapper(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        context_length: int,
        dropout_p: float,
        num_heads: int,
        qkv_bias=False,
    ):
        super().__init__()

        self.multihead_layer = torch.nn.ModuleList(
            [
                MaskedAttention(
                    embd_dim_in=dim_in,
                    dim_out=dim_out,
                    context_length=context_length,
                    dropout=dropout_p,
                    qkv_bias=qkv_bias,
                )
                for i in range(num_heads)
            ]
        )

    def forward(self, batch_inputs):
        return torch.cat(
            [
                masked_attention_layer(batch_inputs)
                for masked_attention_layer in self.multihead_layer
            ],
            dim=-1,
        )


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_dim_out: int,
        num_heads: int,
        context_len: int,
        dropout_p: float,
        qkv_bias=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim_out = head_dim_out
        self.num_heads = num_heads

        self.total_dim_out = head_dim_out * num_heads

        self.W_query = torch.nn.Linear(
            in_features=embed_dim, out_features=self.total_dim_out, bias=qkv_bias
        )
        self.W_key = torch.nn.Linear(
            in_features=embed_dim, out_features=self.total_dim_out, bias=qkv_bias
        )
        self.W_value = torch.nn.Linear(
            in_features=embed_dim, out_features=self.total_dim_out, bias=qkv_bias
        )

        # dropout layer
        self.dropout = torch.nn.Dropout(p=dropout_p)

        # mask
        self.register_buffer(
            name="mask",
            tensor=torch.triu(torch.ones(size=(context_len, context_len)), diagonal=1),
        )

        # output project layer
        ## apparently used in LLMs
        self.output_proj = torch.nn.Linear(self.total_dim_out, self.total_dim_out)

    def forward(self, X):
        """

        :param X: shape: [batch size x num_tokens x embed_dim]
        :return: output project layer; shape: [batch size, num_tokens, num_heads x head_dim]
        """
        batch_size, num_tokens, embed_dim = X.shape

        queries = self.W_query(X)
        keys = self.W_key(X)
        values = self.W_value(X)
        # shape: [batch_size, num_tokens, total_dim_out]

        # reshape
        queries = queries.view(
            batch_size, num_tokens, self.num_heads, self.head_dim_out
        )
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim_out)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim_out)

        # transpose
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # attention score
        attention_scores = queries @ keys.transpose(2, 3)

        ## apply mask
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(mask, -torch.inf)

        # attention weights
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        # dropout
        attention_weights = self.dropout(attention_weights)

        # context vector
        context_vec = (attention_weights @ values).transpose(1, 2)

        # reshape
        context_vec = context_vec.contiguous().view(
            batch_size, num_tokens, self.total_dim_out
        )

        # transform
        context_vec = self.output_proj(context_vec)

        return context_vec


if __name__ == "__main__":
    num_heads = 3
    batch_size = 2
    context_len = 6
    num_tokens = 5
    embed_dim = 5
    context_output_dim = 4
    dropout_prob = 0.5

    torch.manual_seed(100)
    test_batch_inputs = torch.rand(size=(batch_size, num_tokens, embed_dim))

    # multihead_layer = MultiHeadAttentionWrapper(dim_in=embed_dim,
    #                                             dim_out=context_output_dim,
    #                                             context_length=context_len,
    #                                             dropout_p=dropout_prob,
    #                                             num_heads=num_heads,
    #                                             qkv_bias=False)
    # test_outputs = multihead_layer(test_batch_inputs)

    multihead_attention = MultiHeadAttention(
        embed_dim=embed_dim,
        head_dim_out=context_output_dim,
        num_heads=num_heads,
        context_len=context_len,
        dropout_p=dropout_prob,
    )
    test_outputs = multihead_attention(test_batch_inputs)

    print(f"Shape: {test_outputs.shape}")
