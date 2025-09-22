


GPT2_CONFIG = {
    "vocab_size"                   : 50257,        # total numbr of words in english vocab
    "context_length"               : 5,            # total no of tokens in one row of input that LLM can see any point in time
    "embedding_dimension"          : 768,          #  Token representation as vector with 768 columns
    "number_attention_heads"       : 12,           # no of attention head in one transformation block
    "number_neural_network_layers" : 12,           # no of sequential transformation blocks
    "dropout_rate"                 : 0.1,          # drop out rate in the trained neural network
    "qkv_bias"                     : False
}


