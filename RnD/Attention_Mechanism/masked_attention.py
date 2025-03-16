import torch


class MaskedAttention(torch.nn.Module):
    """
    Also referred to as 'Causal Attention'
    """

    def __init__(self, embd_dim_in:int, dim_out:int, context_length:int, dropout:float, qkv_bias=False):

        super().__init__()

        self.W_query = torch.nn.Linear(in_features=embd_dim_in, out_features=dim_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(in_features=embd_dim_in, out_features=dim_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(in_features=embd_dim_in, out_features=dim_out, bias=qkv_bias)

        self.register_buffer(name='mask',
                             tensor=torch.triu(input=torch.ones(size=(context_length,
                                                                      context_length)),
                                               diagonal=1)
                             )
        self.dropout_layer = torch.nn.Dropout(p=dropout)

    def forward(self, batch_inputs: torch.Tensor):

        batch_size, num_tokens, embed_dim = batch_inputs.shape

        batch_queries=self.W_query(batch_inputs)
        batch_keys=self.W_key(batch_inputs)
        batch_values=self.W_value(batch_inputs)

        batch_attn_scores = batch_queries @ batch_keys.transpose(1,2)

        # masking in-place
        batch_attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # num_tokens can be less than context_length
        # context_length is the cut-off

        batch_attn_weights = torch.softmax(batch_attn_scores/batch_keys.shape[-1]**0.5, dim=-1)

        # apply dropout
        batch_attn_weights = self.dropout_layer(batch_attn_weights)

        # context vectors
        z_context_vec = batch_attn_weights @ batch_values

        return z_context_vec


if __name__=="__main__":

    torch.manual_seed(123)

    cntxt_len = 6
    num_tokens = 5
    embed_dim = 3
    test_batch = torch.rand(size=(2, num_tokens, embed_dim))

    p_dropout = 0.5
    test_attention_layer = MaskedAttention(embd_dim_in=embed_dim,
                                           dim_out=5, context_length=cntxt_len, dropout=p_dropout)

    test_batch_output = test_attention_layer(test_batch)
    print(test_batch_output)
    print(f"Shape: {test_batch_output.shape}")
