import torch
from masked_attention import MaskedAttention

class MultiHeadAttentionWrapper(torch.nn.Module):

    def __init__(self, dim_in:int, dim_out:int, context_length:int, dropout_p:float, num_heads:int, qkv_bias=False):

        super().__init__()

        self.multihead_layer = torch.nn.ModuleList(
            [MaskedAttention(embd_dim_in=dim_in,
                             dim_out=dim_out,
                             context_length=context_length,
                             dropout=dropout_p,
                             qkv_bias=qkv_bias)
             for i in range(num_heads)
             ]
        )

    def forward(self, batch_inputs):

        return torch.cat([masked_attention_layer(batch_inputs)
                          for masked_attention_layer in self.multihead_layer
                          ], dim=-1
                         )

if __name__=="__main__":

    num_heads = 3
    batch_size = 2
    context_len = 6
    num_tokens = 5
    embed_dim = 5
    context_output_dim = 4
    dropout_prob = 0.5

    torch.manual_seed(100)
    test_batch_inputs = torch.rand(size=(batch_size,num_tokens, embed_dim))

    multihead_layer = MultiHeadAttentionWrapper(dim_in=embed_dim,
                                                dim_out=context_output_dim,
                                                context_length=context_len,
                                                dropout_p=dropout_prob,
                                                num_heads=num_heads,
                                                qkv_bias=False)

    test_outputs = multihead_layer(test_batch_inputs)

    print(f"Shape: {test_outputs.shape}")
