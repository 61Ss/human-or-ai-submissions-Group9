"""Pretrained-model features: GPT-2 surprisal statistics and MiniLM sentence embeddings.

Weights come from the Hugging Face Hub (gpt2, sentence-transformers/all-MiniLM-L6-v2) and are used
only as fixed feature extractors. Nothing here sees labels.
"""
import os

# Fail with an out-of-memory error instead of pushing the whole machine into swap.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")  # must not exceed the high ratio

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def free_device_memory():
    """MPS keeps freed buffers cached unless told otherwise."""
    if DEVICE == "mps":
        torch.mps.empty_cache()


def _batches(texts, batch_size):
    """Yield (positions, texts), longest first, so padding is small."""
    order = np.argsort([-len(t) for t in texts])
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        yield idx, [texts[i] for i in idx]


@torch.no_grad()
def surprisal_features(texts, model_name="gpt2", max_length=512, batch_size=8):
    """Summaries of per-token surprisal: AI text is often more predictable to a language model."""
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE).eval()
    rows = [None] * len(texts)
    for idx, chunk in _batches(texts, batch_size):
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(DEVICE)
        logp = model(**enc).logits[:, :-1].log_softmax(-1)  # about 0.8 GB per batch at full length
        target = logp.gather(-1, enc.input_ids[:, 1:].unsqueeze(-1))
        nll = (-target.squeeze(-1)).cpu().numpy()
        rank = (logp > target).sum(-1).cpu().numpy()
        mask = enc.attention_mask[:, 1:].bool().cpu().numpy()
        del logp, target, enc
        free_device_memory()
        for pos, n, r, m in zip(idx, nll, rank, mask):
            n, r = n[m], r[m]
            half = len(n) // 2
            rows[pos] = {
                "gpt2_nll_mean": n.mean(), "gpt2_nll_std": n.std(),
                "gpt2_nll_q10": np.quantile(n, 0.1), "gpt2_nll_q50": np.quantile(n, 0.5),
                "gpt2_nll_q90": np.quantile(n, 0.9),
                "gpt2_nll_first_half": n[:half].mean(), "gpt2_nll_second_half": n[half:].mean(),
                "gpt2_top1": (r == 0).mean(), "gpt2_top10": (r < 10).mean(), "gpt2_top100": (r < 100).mean(),
                "gpt2_rare": (r >= 1000).mean(),
                "gpt2_diff_std": np.diff(n).std() if len(n) > 1 else 0.0,
            }
    return pd.DataFrame(rows)


@torch.no_grad()
def curvature_features(texts, model_name="gpt2", max_length=512, batch_size=8):
    """Analytic conditional probability curvature, following Fast-DetectGPT (Bao et al., 2024).

    Each token's log-probability is compared with its expectation under the model's own predictive
    distribution, which adjusts for how uncertain the model was at that position.
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE).eval()
    rows = [None] * len(texts)
    for idx, chunk in _batches(texts, batch_size):
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(DEVICE)
        logp = model(**enc).logits[:, :-1].log_softmax(-1)
        prob = logp.exp()
        expected = (prob * logp).sum(-1)
        variance = (prob * logp.square()).sum(-1) - expected.square()
        observed = logp.gather(-1, enc.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        gap, mu, var = (observed - expected).cpu().numpy(), expected.cpu().numpy(), variance.clamp_min(1e-6).cpu().numpy()
        mask = enc.attention_mask[:, 1:].bool().cpu().numpy()
        del logp, prob, expected, variance, observed, enc
        free_device_memory()
        for pos, g, u, v, m in zip(idx, gap, mu, var, mask):
            g, u, v = g[m], u[m], v[m]
            z = g / np.sqrt(v)
            rows[pos] = {
                "curv_total": g.sum() / np.sqrt(v.sum()), "curv_z_mean": z.mean(), "curv_z_std": z.std(),
                "curv_z_q10": np.quantile(z, 0.1), "curv_z_q90": np.quantile(z, 0.9),
                "curv_entropy_mean": -u.mean(), "curv_entropy_std": u.std(),
            }
    return pd.DataFrame(rows)


@torch.no_grad()
def sentence_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", max_length=256, batch_size=64):
    """Mean-pooled, L2-normalized token embeddings."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for idx, chunk in _batches(texts, batch_size):
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(DEVICE)
        hidden = model(**enc).last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        out[idx] = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()
    return out
