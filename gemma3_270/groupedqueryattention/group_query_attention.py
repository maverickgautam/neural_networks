import torch.nn as nn
import torch
from gemma3_270.gemma3_270_model.repo import Rope
from gemma3_270.groupedqueryattention.RMSNorm import RMSNorm


class GroupQueryAttention(nn.Module):

    def __init__(self,d_in, num_heads, num_kv_group, head_dim=None, qk_norm=False, query_pre_attn_scalar=None, dType=None):

        super().__init__()
        # num_heads divisible by num_kv_group

        assert  num_heads % num_kv_group == 0
        self.num_heads = num_heads
        self.num_kv_group = num_kv_group
        self.group_size = self.num_heads // self.num_kv_group

        if head_dim is None:
            assert d_in % num_heads == 0
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim


        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dType )
        self.W_key   = nn.Linear(d_in, self.num_kv_group * head_dim, bias=False, dtype=dType)
        self.W_value = nn.Linear(d_in, self.num_kv_group * head_dim, bias=False, dtype=dType)
        self.out_proj = nn.Linear( self.d_out, d_in, bias=False, dtype=dType)
        self.rope = Rope()


        if qk_norm:
            self.q_norm = RMSNorm(head_dim,eps=1e-6)
            self.k_norm = RMSNorm(head_dim,eps=1e-6)
        else:
            self.q_norm = self.k_norm = None


        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar ** -0.5

        else:
            self.scaling = head_dim ** -0.5



        def forward(self, x, mask, cos,sin):
            b, num_tokens, _ = x.shape

            # Apply Projections
            queries = self.W_query(x)
            keys = self.W_key(x)
            values = self.W_value(x)

            # Reshape

            queries = queries.view(b, num_tokens,self.num_heads, self.head_dim ).transpose(1, 2)
            keys = keys.view(b, num_tokens, self.num_kv_group, self.head_dim ).transpose(1, 2)
            values = values.view(b, num_tokens, self.num_kv_group, self.head_dim ).transpose(1, 2)

            #optimal Normalization

            if self.q_norm:
                queries = self.q_norm(queries)

            if self.k_norm:
                keys = self.k_norm(keys)

            # Apply Rope
            queries = self.rope.apply_rope(queries, cos, sin)
            keys    = self.rope.apply_rope(keys, cos, sin)

            # Expand K and V to match number of Heads
            keys = keys.repeat_interleave(self.group_size, dim=1)
            values = values.repeat_interleave(self.group_size, dim=1)

            # Scale Queries
            queries = queries * self.scaling

            # Attention
            attn_score = queries @ keys.transpose(2, 3)
            attn_score = attn_score.masked_fill(mask, -torch.inf)
            attn_weight = torch.softmax(attn_score, dim=-1)

            # Context Vector
            context = ( attn_weight @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
            return  self.out_proj(context)
