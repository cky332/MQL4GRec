"""Pointwise candidate scorer for the 1-pos+20-neg protocol.

MQL4GRec only does constrained beam search; it has no pointwise scorer. Here we
compute, for a user history and a candidate item's code, the T5 sequence
log-likelihood under teacher forcing (length-normalized, i.e. the same semantics
as beam search's `sequences_scores`). This lets us rank a small candidate set the
way MLLM-MSR ranks 21 candidates by P(Yes).
"""
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import _common as C


class Scorer:
    def __init__(self, ckpt_path, device=None, max_his_len=20, max_len=512):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = T5Tokenizer.from_pretrained(ckpt_path)
        self.model = T5ForConditionalGeneration.from_pretrained(ckpt_path).to(self.device).eval()
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.max_his_len = max_his_len
        self.max_len = max_len

    def history_input(self, history_ids, index):
        """Concatenate the history items' codes (no separators, add_prompt=False)."""
        hist = history_ids[-self.max_his_len:] if self.max_his_len > 0 else history_ids
        return "".join(C.code_str(index[str(int(i))]) for i in hist)

    @torch.no_grad()
    def score(self, input_str, candidate_strs):
        """Mean per-token log-prob of each candidate code given the history.

        Returns a list of floats (higher = more likely), comparable across
        candidates and across channels (all candidate codes have equal length).
        """
        enc = self.tok([input_str], return_tensors="pt", truncation=True,
                       max_length=self.max_len)
        lab = self.tok(candidate_strs, return_tensors="pt", padding="longest").input_ids
        n = lab.size(0)
        input_ids = enc.input_ids.to(self.device).repeat(n, 1)
        attn = enc.attention_mask.to(self.device).repeat(n, 1)
        lab = lab.to(self.device)
        labels_in = lab.clone()
        labels_in[lab == self.pad] = -100
        logits = self.model(input_ids=input_ids, attention_mask=attn, labels=labels_in).logits
        logp = logits.log_softmax(-1)
        mask = lab.ne(self.pad)
        gathered = logp.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
        return (gathered.sum(1) / mask.sum(1).clamp(min=1)).float().cpu().tolist()
