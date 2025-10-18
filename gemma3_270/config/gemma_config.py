import torch

GEMMA270_CONFIG = {
    "vocab_size"                   : 50257,        # total numbr of words in english vocab
    "context_length"               : 32_768,            # total no of tokens in one row of input that LLM can see any point in time
    "embedding_dimension"          : 640,          #  Token representation as vector with 768 columns
    "number_attention_heads"       : 4,           # no of attention head in one transformation block
    "number_neural_network_layers" : 18,           # no of sequential transformation blocks
    "hidden_dim"                   : 256,
    "head_dim"                     : 256,
    "qk_norm"                      : True,
    "n_kv_groups"                  : 1,
    "rope_local_base"              : 10_000.00,
    "rope_base"                    : 1_000_000.0,
    "dropout_rate"                 : 0.1,          # drop out rate in the trained neural network
    "qkv_bias"                     : False,
    "number_of_kv_group"           : 10,
    "sliding_window"               : 512,
    "layer_types"                  : [
        "sliding_attentions",
        "sliding_attentions",
        "sliding_attentions",
        "sliding_attentions",
        "sliding_attentions",
        "full_attention",
        "sliding_attentions",
        "sliding_attentions",
        "sliding_attentions",
        "sliding_attentions",
        "sliding_attentions",
        "full_attention",
        "sliding_attentions",
        "sliding_attentions",
        "sliding_attentions",
        "sliding_attentions",
        "sliding_attentions",
        "full_attention",
    ],
    "dtype"                      : torch.bfloat16,
}


