"""
Reward functions for Stage 2 PPO, matching TEAR paper Eq. 3 and Eq. 5.

R_pmt (prompt-level, Eq. 3):
    R_pmt(p_t) = alpha1 * (1 - g_t(p_t)) + alpha2 * (g_r(p_t) + 1) / 2
        g_t : hate-speech classifier confidence on the rewritten prompt p_t
        g_r : cosine similarity between p_t and the average embedding of
              reference temporal-style prompts (the prototype)

R_con (temporal consistency, Eq. 5):
    R_con(p_s, p_t) = min(beta,
        (g_gc(p_s, video) - gamma1) / theta1
      + (g_ic(video)       - gamma2) / theta2 )
        g_gc : global video-text consistency between the *original harmful seed*
               p_s and the generated video (TA head of VideoAlign with p_s)
        g_ic : inner video consistency / quality (VQ + MQ of VideoAlign)

Final scalar reward returned to PPO is R_pmt + R_con. The KL penalty against the
reference policy is added by trlx itself via init_kl_coef in ppo_config.yml.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import torch


def _load_reference_prompts(path: str | None, fallback: Iterable[str] | None = None) -> list[str]:
    """Load reference temporal-style prompts used to build the prototype.

    Accepts:
      - .txt : one prompt per line
      - .json/.jsonl with objects containing an 'output' or 'prompt' or 'text' field
    """
    if path and os.path.exists(path):
        prompts: list[str] = []
        if path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                prompts = [ln.strip() for ln in f if ln.strip()]
        else:
            items: list = []
            if path.endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        items.append(json.loads(line))
            else:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else data.get("data", [])
            for obj in items:
                if isinstance(obj, str):
                    prompts.append(obj.strip())
                    continue
                for key in ("output", "prompt", "text", "rewritten", "target"):
                    val = obj.get(key) if isinstance(obj, dict) else None
                    if isinstance(val, str) and val.strip():
                        prompts.append(val.strip())
                        break
        prompts = [p for p in prompts if p]
        if prompts:
            return prompts

    if fallback:
        fb = [p for p in fallback if isinstance(p, str) and p.strip()]
        if fb:
            return fb
    return []


def _uniform_sample_prompts(prompts: list[str], sample_count: int | None) -> tuple[list[str], list[int]]:
    """Deterministically pick evenly spaced reference prompts."""
    if not sample_count or sample_count <= 0 or sample_count >= len(prompts):
        return prompts, list(range(len(prompts)))
    if sample_count == 1:
        return [prompts[0]], [0]

    last = len(prompts) - 1
    raw_indices = [round(i * last / (sample_count - 1)) for i in range(sample_count)]
    indices: list[int] = []
    used: set[int] = set()
    for raw in raw_indices:
        idx = min(max(int(raw), 0), last)
        if idx in used:
            right = idx + 1
            while right <= last and right in used:
                right += 1
            left = idx - 1
            while left >= 0 and left in used:
                left -= 1
            idx = right if right <= last else left
        if idx < 0 or idx in used:
            continue
        used.add(idx)
        indices.append(idx)
    indices = sorted(indices)
    return [prompts[i] for i in indices], indices


class PromptReward:
    """Implements R_pmt from Eq. 3 of the paper."""

    def __init__(
        self,
        hate_model_id: str,
        sentence_encoder_id: str,
        reference_prompts: list[str],
        device: str = "cpu",
        alpha1: float = 0.5,
        alpha2: float = 0.5,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self._available = True

        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                AutoModel,
            )
        except Exception as e:
            print(f"[PromptReward] transformers import failed: {e}")
            self._available = False
            return

        try:
            self.hate_tokenizer = AutoTokenizer.from_pretrained(hate_model_id)
            self.hate_model = AutoModelForSequenceClassification.from_pretrained(
                hate_model_id, torch_dtype=dtype
            ).to(device).eval()
            self.hate_label_to_idx = self._infer_hate_label_index(self.hate_model.config)
        except Exception as e:
            print(f"[PromptReward] hate-speech classifier load failed ({hate_model_id}): {e}")
            self.hate_model = None

        try:
            self.enc_tokenizer = AutoTokenizer.from_pretrained(sentence_encoder_id)
            self.enc_model = AutoModel.from_pretrained(
                sentence_encoder_id, torch_dtype=dtype
            ).to(device).eval()
        except Exception as e:
            print(f"[PromptReward] sentence encoder load failed ({sentence_encoder_id}): {e}")
            self.enc_model = None

        self.prototype = None
        if self.enc_model is not None and reference_prompts:
            with torch.no_grad():
                embs = self._encode(reference_prompts, batch_size=16)
                proto = embs.mean(dim=0)
                proto = proto / (proto.norm(p=2) + 1e-12)
                self.prototype = proto
            print(f"[PromptReward] prototype built from {len(reference_prompts)} reference prompts.")
        else:
            print("[PromptReward] no prototype (encoder or reference prompts unavailable); g_r set to 0.")

    @staticmethod
    def _infer_hate_label_index(config) -> int:
        id2label = getattr(config, "id2label", None) or {}
        for i, name in id2label.items():
            if isinstance(name, str) and ("hate" in name.lower() or "toxic" in name.lower() or name.lower() == "label_1"):
                return int(i)
        return 1 if getattr(config, "num_labels", 2) >= 2 else 0

    def _encode(self, texts: list[str], batch_size: int = 8) -> torch.Tensor:
        """Mean-pooled sentence embeddings, L2-normalised per sentence."""
        all_vecs: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            toks = self.enc_tokenizer(
                chunk, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(self.device)
            out = self.enc_model(**toks)
            hidden = out.last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            vecs = summed / counts
            vecs = vecs / (vecs.norm(p=2, dim=-1, keepdim=True) + 1e-12)
            all_vecs.append(vecs.detach().float().cpu())
        return torch.cat(all_vecs, dim=0)

    def _toxicity(self, texts: list[str]) -> list[float]:
        if self.hate_model is None:
            return [0.0] * len(texts)
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(texts), 8):
                chunk = texts[start:start + 8]
                toks = self.hate_tokenizer(
                    chunk, padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(self.device)
                logits = self.hate_model(**toks).logits.float()
                probs = torch.softmax(logits, dim=-1)
                idx = min(self.hate_label_to_idx, probs.shape[-1] - 1)
                scores.extend(probs[:, idx].cpu().tolist())
        return scores

    def _similarity(self, texts: list[str]) -> list[float]:
        if self.prototype is None or self.enc_model is None:
            return [0.0] * len(texts)
        with torch.no_grad():
            embs = self._encode(texts, batch_size=8)
            sims = (embs @ self.prototype.float()).clamp(-1.0, 1.0)
        return sims.tolist()

    def __call__(self, rewritten_prompts: list[str]) -> list[float]:
        if not self._available:
            return [0.0] * len(rewritten_prompts)
        tox = self._toxicity(rewritten_prompts)
        sim = self._similarity(rewritten_prompts)
        rewards: list[float] = []
        for gt, gr in zip(tox, sim):
            r = self.alpha1 * (1.0 - float(gt)) + self.alpha2 * (float(gr) + 1.0) / 2.0
            rewards.append(float(r))
        return rewards


class TemporalReward:
    """Implements R_con from Eq. 5 of the paper.

    Wraps a VideoAlign-style reward inferencer that returns per-video dicts with
    keys {'VQ','MQ','TA'} (text alignment uses the provided prompt).
        g_gc = TA evaluated against the original seed p_s
        g_ic = VQ + MQ (video-only quality, prompt-independent)
    """

    def __init__(
        self,
        reward_inferencer,
        gamma1: float = 0.0,
        gamma2: float = 0.0,
        theta1: float = 1.0,
        theta2: float = 1.0,
        beta: float = 5.0,
    ):
        self.inferencer = reward_inferencer
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)
        self.theta1 = max(float(theta1), 1e-6)
        self.theta2 = max(float(theta2), 1e-6)
        self.beta = float(beta)

    def score(self, video_paths: list[str], seed_prompts: list[str]) -> tuple[list[float], list[dict]]:
        """Return (rewards, raw_reward_dicts)."""
        try:
            raw = self.inferencer.reward(
                video_paths=video_paths, prompts=seed_prompts, use_norm=True
            )
        except Exception as e:
            print(f"[TemporalReward] reward inference failed: {e}")
            raw = [{"VQ": 0.0, "MQ": 0.0, "TA": 0.0, "Overall": 0.0} for _ in video_paths]

        rewards: list[float] = []
        for d in raw:
            ggc = float(d.get("TA", 0.0))
            gic = float(d.get("VQ", 0.0)) + float(d.get("MQ", 0.0))
            val = (ggc - self.gamma1) / self.theta1 + (gic - self.gamma2) / self.theta2
            rewards.append(min(self.beta, val))
        return rewards, raw


def build_prompt_reward_from_config(cfg, fallback_prompts: list[str] | None = None,
                                    device: str = "cpu") -> PromptReward | None:
    """Construct PromptReward from the 'prompt_reward' section of cfg.video."""
    pr_cfg = getattr(cfg.video, "prompt_reward", None) or {}
    if isinstance(pr_cfg, dict):
        enabled = pr_cfg.get("enabled", True)
    else:
        enabled = getattr(pr_cfg, "enabled", True)
        pr_cfg = {
            "hate_model_id": getattr(pr_cfg, "hate_model_id", None),
            "sentence_encoder_id": getattr(pr_cfg, "sentence_encoder_id", None),
            "reference_prompts_path": getattr(pr_cfg, "reference_prompts_path", None),
            "reference_sample_count": getattr(pr_cfg, "reference_sample_count", None),
            "alpha1": getattr(pr_cfg, "alpha1", 0.5),
            "alpha2": getattr(pr_cfg, "alpha2", 0.5),
        }
    if not enabled:
        return None

    hate_id = pr_cfg.get("hate_model_id") or "facebook/roberta-hate-speech-dynabench-r4-target"
    enc_id = pr_cfg.get("sentence_encoder_id") or "sentence-transformers/all-MiniLM-L6-v2"
    ref_path = pr_cfg.get("reference_prompts_path")
    ref_prompts = _load_reference_prompts(ref_path, fallback=fallback_prompts)
    raw_sample_count = pr_cfg.get("reference_sample_count")
    try:
        sample_count = int(raw_sample_count) if raw_sample_count is not None else None
    except (TypeError, ValueError):
        sample_count = None
    if ref_prompts and sample_count:
        ref_prompts, ref_indices = _uniform_sample_prompts(ref_prompts, sample_count)
        print(
            f"[PromptReward] uniformly sampled {len(ref_prompts)} reference prompts "
            f"from {ref_path or 'fallback'} at indices {ref_indices}."
        )
    return PromptReward(
        hate_model_id=hate_id,
        sentence_encoder_id=enc_id,
        reference_prompts=ref_prompts,
        device=device,
        alpha1=float(pr_cfg.get("alpha1", 0.5)),
        alpha2=float(pr_cfg.get("alpha2", 0.5)),
    )


def build_temporal_reward_from_config(cfg, reward_inferencer) -> TemporalReward:
    tr_cfg = getattr(cfg.video, "temporal_reward", None) or {}
    if not isinstance(tr_cfg, dict):
        tr_cfg = {
            "gamma1": getattr(tr_cfg, "gamma1", 0.0),
            "gamma2": getattr(tr_cfg, "gamma2", 0.0),
            "theta1": getattr(tr_cfg, "theta1", 1.0),
            "theta2": getattr(tr_cfg, "theta2", 1.0),
            "beta": getattr(tr_cfg, "beta", 5.0),
        }
    return TemporalReward(
        reward_inferencer=reward_inferencer,
        gamma1=float(tr_cfg.get("gamma1", 0.0)),
        gamma2=float(tr_cfg.get("gamma2", 0.0)),
        theta1=float(tr_cfg.get("theta1", 1.0)),
        theta2=float(tr_cfg.get("theta2", 1.0)),
        beta=float(tr_cfg.get("beta", 5.0)),
    )
