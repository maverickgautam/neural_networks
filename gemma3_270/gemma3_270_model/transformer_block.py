from torch._C.cpp import nn
from gemma3_270.config.gemma_config import GEMMA270_CONFIG
from gemma3_270.groupedqueryattention.RMSNorm import RMSNorm
from gemma3_270.groupedqueryattention.group_query_attention import GroupQueryAttention
from llm_gpt.gpt_model.chatgpt import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, attn_type:str):
        super().__init__()
        self.cfg = GEMMA270_CONFIG
        self.attn_type = attn_type

        self.attn = GroupQueryAttention(
            d_in=GEMMA270_CONFIG["embedding_dim"],
            num_heads=GEMMA270_CONFIG["num_heads"],
            num_kv_group=GEMMA270_CONFIG["num_kv_group"],
            head_dim=GEMMA270_CONFIG["head_dim"],
            qk_norm=GEMMA270_CONFIG["qk_norm"],
            query_pre_attn_scalar=GEMMA270_CONFIG["query_pre_attn_scalar"],
            dType=GEMMA270_CONFIG["dtype"],
        )

        self.ff = FeedForward()
        self.input_layernorm = RMSNorm(GEMMA270_CONFIG["embedding_dim"],eps=1e-6)
        self.post_attn_layerednorm = RMSNorm(GEMMA270_CONFIG["embedding_dim"],eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm(GEMMA270_CONFIG["embedding_dim"],eps=1e-6)
        self.post_feedforward_layernorm = RMSNorm(GEMMA270_CONFIG["embedding_dim"],eps=1e-6)


        def forward(self,
                    x,
                    mask_global,
                    mask_local,
                    cos_global,
                    sin_global,
                    cos_local,
                    sin_local,):

            shortcut = x
            x = self.input_layernorm(x)

            if self.attn_type == "sliding_attention":
                attn_mask = mask_local
                cos = cos_local
                sin = sin_local
            else :
                attn_mask = mask_global
                cos = cos_global
                sin = sin_global

            x_attn = self.attn(x, attn_mask=attn_mask, cos=cos, sin=sin)
            x_attn = self.post_attn_layerednorm(x_attn)
            x = shortcut + x_attn

            shortcut = x
            x_ffn = self.pre_feedforward_layernorm(x)
            x_ffn = self.ff(x_ffn)
            x_ffn = self.post_feedforward_layernorm(x_ffn)
            x= shortcut + x_ffn
            return x
