import torch.nn as nn
import torch
from gemma3_270.config.gemma_config import GEMMA270_CONFIG
from gemma3_270.gemma3_270_model.repo import Rope
from gemma3_270.gemma3_270_model.transformer_block import TransformerBlock
from gemma3_270.groupedqueryattention.RMSNorm import RMSNorm


class Gemma270(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert GEMMA270_CONFIG["layer_type"] is not None and len(GEMMA270_CONFIG["layer_type"]) == GEMMA270_CONFIG["n_layers"]

        # Main model Params
        self.token_emb = nn.Embedding(GEMMA270_CONFIG["vocab_size"], GEMMA270_CONFIG["embedding_dim"], dtype=GEMMA270_CONFIG["dtype"])
        self.block = nn.ModuleList([TransformerBlock[GEMMA270_CONFIG, attn_type] for attn_type in GEMMA270_CONFIG["layer_types"]])

        self.final_norm = RMSNorm(GEMMA270_CONFIG["embedding_dim"], eps=1e-6)
        self.out_head = nn.Linear(GEMMA270_CONFIG["embedding_dim"], GEMMA270_CONFIG["vocab_size"], bias=False, dtype=GEMMA270_CONFIG["dtype"])

        self.cfg = GEMMA270_CONFIG
        self.rope = Rope()

        cos_local, sin_local = self.rope.compute_rope_param(head_dim=GEMMA270_CONFIG["head_dim"], theta_base=GEMMA270_CONFIG["theta_base"],
                                                            context_length=GEMMA270_CONFIG["context_length"], dtype=torch.float32)

        cos_global, sin_global = self.rope.compute_rope_param(head_dim=GEMMA270_CONFIG["head_dim"], theta_base=GEMMA270_CONFIG["theta_base"],
                                                              context_length=GEMMA270_CONFIG["context_length"], dtype=torch.float32)

        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def create_mask(self, seq_len, device):
        ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        # set elements to one for  all elements up of the diagonal (top right hand side )
        mask_global = torch.triu(ones, diagonal=1)
        far_past = torch.triu(ones, diagonal=self.cfg["sliding_window"])
        mask_local = mask_global | far_past
        return mask_global, mask_local

    def forward(self, input_ids, targets=None):
        b, seq_len = input_ids.shape
        x = self.token_emb(input_ids) * (self.cfg["embedding_dim"] ** 0.5)
        mask_global, mask_local = self.create_mask(seq_len, x.device)

        for block in self.block:
            x = block(
                x,
                mask_global=mask_global,
                mask_local=mask_local,
                cos_global=self.cos_global,
                sin_global=self.sin_global,
                cos_local=self.cos_local,
                sin_local=self.sin_local,
            )

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))

        loss = None
        if targets is not None:
            loss = torch.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            ctx_len = self.cfg["context_len"]
            idx_cond = idx if idx.size(1) <= ctx_len else idx[:-ctx_len]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=0)

        return idx
