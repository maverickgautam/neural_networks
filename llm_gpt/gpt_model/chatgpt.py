#124 M parameters of GPT2
import math
import torch
import torch.nn as nn
from llm_gpt.config.gpt_config import GPT2_CONFIG


class LayerNorm(nn.Module):

    def __init__(self):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(GPT2_CONFIG["embedding_dimension"]))
        self.shift = nn.Parameter(torch.zeros(GPT2_CONFIG["embedding_dimension"]))

    def forward(self, inputs):

        mean = inputs.mean(-1, keepdim=True)
        variance = inputs.var(-1, keepdim=True, unbiased=False)
        normalized_input = (inputs - mean) / torch.sqrt(variance + self.eps)
        return self.scale * normalized_input + self.shift



class Gelu(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return 0.5 * inputs * (1 + torch.tanh(
            torch.sqrt( torch.tensor(2.0/torch.pi)) *
            (inputs + .044715 * torch.pow(inputs, 3))))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(GPT2_CONFIG["embedding_dimension"], 4*GPT2_CONFIG["number_attention_heads"]),
            Gelu(),
            nn.Linear(4*GPT2_CONFIG["number_attention_heads"],GPT2_CONFIG["embedding_dimension"])
        )

    def forward(self, inputs):
        return self.layers(inputs)


class MultiHeadAttention(nn.Module):

    # num_heads = Number of attentionion head - parallel heads
    # for GPT d_output and d_input will be the same
    def __init__(self, d_input, d_output, dropout, num_heads):
        super().__init__()
        assert d_output % num_heads == 0

        self.d_output  = d_output
        self.num_heads = num_heads

        # nuber of columns in a single attention head - this is a subset of embedding_dimension
        self.head_dim  = d_output // num_heads

        self.Query     = nn.Linear(d_input, d_output , bias=GPT2_CONFIG["qkv_bias"] )
        self.Key       = nn.Linear(d_input, d_output,  bias=GPT2_CONFIG["qkv_bias"])
        self.Value     = nn.Linear(d_input, d_output,  bias=GPT2_CONFIG["qkv_bias"])
        self.out_proj  = nn.Linear(d_output, d_output)
        self.dropout   = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones( GPT2_CONFIG["context_length"],
                            GPT2_CONFIG["context_length"]), diagonal=1))


    def forward(self, inputs):
        b, num_tokens, d_in= inputs.shape


        keys    = self.Key(inputs)
        queries = self.Query(inputs)
        values  = self.Value(inputs)
        # The dimensions  is being increased from 3 to 4
        keys    =   keys.view(b,  num_tokens, self.num_heads, self.head_dim)
        queries =   queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values  =   values.view(b, num_tokens, self.num_heads, self.head_dim)

        # for a given batch group by Heads.  in a head you have tokens as num rows and head_dim as number of column
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # enable mtrix multiplication and generates batch, num_heads, num_tokens, num_tokens
        attn_score = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_score.masked_fill_(mask_bool, -torch.inf)

        attn_weight = torch.softmax(attn_score/keys.shape[-1]**0.5, dim=-1)
        attn_weight = self.dropout(attn_weight)

        context_vector = (attn_weight @ values).transpose(1, 2)

        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_output)
        context_vector = self.out_proj(context_vector)

        return context_vector



class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            GPT2_CONFIG["embedding_dimension"],
            GPT2_CONFIG["embedding_dimension"],
            GPT2_CONFIG["dropout_rate"],
            GPT2_CONFIG["number_attention_heads"]
        )
        self.feedforward = FeedForward(config)
        self.layer_norm1 = LayerNorm()
        self.layer_norm2 = LayerNorm()
        self.dropout = nn.Dropout(GPT2_CONFIG["dropout_rate"])


    def forward(self, inputs):

        shortcut = inputs
        inputs = self.layer_norm1(inputs)
        inputs = self.attention(inputs)
        inputs = self.dropout(inputs)
        inputs = inputs + shortcut

        shortcut = inputs
        inputs = self.layer_norm2(inputs)
        inputs = self.feedforward(inputs)
        inputs = self.dropout(inputs)
        inputs = inputs + shortcut

        return inputs


class GPT2 (nn.Module):

    def __init__(self, config):
        super().__init__()
        # Unlike one-hot encoding, which uses a sparse vector to represent each category, embeddings use a dense vector,
        # where each element is a floating-point number and the vector has lower dimensions.

        # GPT2_CONFIG["vocab_size"]          Number of unique categories (or tokens). This is the size of the lookup table.
        # GPT2_CONFIG["embedding_dimension"] The size of the embedding vector for each category.

        self.token_embedding =  nn.Embedding(GPT2_CONFIG["vocab_size"], GPT2_CONFIG["embedding_dimension"])
        self.pos_embedding   =  nn.Embedding(GPT2_CONFIG["context_length"], GPT2_CONFIG["embedding_dimension"])

        self.drop_embedding  =  nn.Dropout(GPT2_CONFIG["dropout_rate"])
        self.transformer = nn.Sequential(*[Transformer(config) for _ in range(GPT2_CONFIG["number_attention_heads"])])

        self.final_normalization = LayerNorm()
        self.output_head = nn.Linear(GPT2_CONFIG["embedding_dimension"], GPT2_CONFIG["vocab_size"], bias=False)


        # Use a placeholder for layerNorm





    def forward(self, in_idx):

        batch_size, seq_len = in_idx.shape
        token_embed = self.token_embedding(in_idx)
        #print("token_embed : ", token_embed)
        pos_embed = self.pos_embedding(torch.arange(seq_len, device=in_idx.device))
        #print("pos_embed :", pos_embed)
        input_embedding = token_embed + pos_embed
        #print("input_embedding : ", input_embedding)
        input_embedding = self.drop_embedding(input_embedding)
        #print("drop  : ", input_embedding)
        input_embedding = self.transformer(input_embedding)
        #print("transformer  : ", input_embedding)
        input_embedding = self.final_normalization(input_embedding)
        #print("final_normalization   : ", input_embedding)
        logits = self.output_head(input_embedding)
        #print("logits   : ", logits)
        return logits

if __name__ == '__main__':

    torch.manual_seed(123)
    x = torch.randn((2, 4,768))
    #tranform  =  Transformer(GPT2_CONFIG)
    #output = tranform(x)
    #print(x.shape)
    #print(output.shape)
    model = GPT2(GPT2_CONFIG)

    # save the model and load the model
    torch.save(model.state_dict(), '../model.pt')
    model.load_state_dict(torch.load('../model.pt'))
    model.eval()

    output = model(x)
    print(x.shape)
    print(output.shape)













