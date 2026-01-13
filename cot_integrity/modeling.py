from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


@dataclass
class ModelBundle:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device


def load_model(model_name: str, device_map: str = "auto", torch_dtype: str = "auto", trust_remote_code: bool = True) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)

    # Some tokenizers have no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = None
    if torch_dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    elif torch_dtype == "bf16":
        dtype = torch.bfloat16
    elif torch_dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust_remote_code)
    model.eval()

    # Best-effort pick a "main device"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return ModelBundle(model=model, tokenizer=tokenizer, device=device)


def build_prompt_input_ids(tokenizer: AutoTokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> torch.Tensor:
    """
    Returns input_ids for chat models using the tokenizer's chat template when available.
    Falls back to a simple concatenation otherwise.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, return_tensors="pt")
    # Fallback: naive prompt
    text = ""
    for m in messages:
        text += f"{m['role'].upper()}: {m['content']}\n"
    if add_generation_prompt:
        text += "ASSISTANT: "
    return tokenizer(text, return_tensors="pt").input_ids


def append_assistant_prefill(tokenizer: AutoTokenizer, base_input_ids: torch.Tensor, assistant_prefill_text: str) -> torch.Tensor:
    """
    Appends raw tokenized assistant-prefill to a prompt that already ends at an assistant generation boundary.
    This is how we do "prefill tampering" without closing <think>.
    """
    pre = tokenizer(assistant_prefill_text, add_special_tokens=False, return_tensors="pt").input_ids
    return torch.cat([base_input_ids, pre], dim=-1)


class StopOnTokenSequences(StoppingCriteria):
    def __init__(self, stop_seqs: list[list[int]]):
        super().__init__()
        self.stop_seqs = [s for s in stop_seqs if len(s) > 0]

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # input_ids: [batch, seq]
        seq = input_ids[0].tolist()
        for stop in self.stop_seqs:
            if len(seq) >= len(stop) and seq[-len(stop) :] == stop:
                return True
        return False


@torch.no_grad()
def generate_assistant_text(
    bundle: ModelBundle,
    messages: list[dict[str, str]],
    assistant_prefill_text: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    skip_special_tokens: bool = False,
    gen_seed: int | None = None,
    stop_strings: list[str] | None = None,
) -> tuple[str, str]:
    """
    Returns (assistant_text, full_decoded_text).
    assistant_text is the newly-generated continuation after prompt(+prefill).

    New:
      - gen_seed: reproducible per-call sampling (important when temperature>0)
      - stop_strings: early stop when a tokenized stop string appears (e.g. </modified>)
    """
    model, tokenizer = bundle.model, bundle.tokenizer
    base_ids = build_prompt_input_ids(tokenizer, messages, add_generation_prompt=True)

    if assistant_prefill_text is not None:
        input_ids = append_assistant_prefill(tokenizer, base_ids, assistant_prefill_text)
    else:
        input_ids = base_ids

    input_ids = input_ids.to(model.device)
    attn = torch.ones_like(input_ids)

    do_sample = temperature > 0.0

    restore_torch_cpu = None
    restore_torch_cuda = None
    if do_sample and gen_seed is not None:
        restore_torch_cpu = torch.random.get_rng_state()
        if torch.cuda.is_available():
            restore_torch_cuda = torch.cuda.get_rng_state_all()

        torch.manual_seed(int(gen_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(gen_seed))

    stopping_criteria = None
    if stop_strings:
        stop_seqs = [tokenizer(s, add_special_tokens=False).input_ids for s in stop_strings]
        stopping_criteria = StoppingCriteriaList([StopOnTokenSequences(stop_seqs)])

    try:
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    finally:
        if restore_torch_cpu is not None:
            torch.random.set_rng_state(restore_torch_cpu)
        if restore_torch_cuda is not None:
            torch.cuda.set_rng_state_all(restore_torch_cuda)

    prompt_len = input_ids.shape[-1]
    new_tokens = gen[0, prompt_len:]
    assistant_text = tokenizer.decode(new_tokens, skip_special_tokens=skip_special_tokens)
    full_text = tokenizer.decode(gen[0], skip_special_tokens=skip_special_tokens)
    return assistant_text, full_text


@torch.no_grad()
def hidden_states_at_last_token(
    bundle: ModelBundle, messages: list[dict[str, str]], assistant_prefill_text: str | None, layers: Sequence[int]
) -> np.ndarray:
    """
    Returns concatenated hidden states at the last token of prompt(+prefill) for selected layers.
    layers are indices into the returned hidden_states tuple, so -1 means final layer output.
    """
    model, tokenizer = bundle.model, bundle.tokenizer
    base_ids = build_prompt_input_ids(tokenizer, messages, add_generation_prompt=True)
    if assistant_prefill_text is not None:
        input_ids = append_assistant_prefill(tokenizer, base_ids, assistant_prefill_text)
    else:
        input_ids = base_ids

    input_ids = input_ids.to(model.device)
    attn = torch.ones_like(input_ids)

    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple: (embeddings, layer1, ..., layerN)
    vecs = []
    for li in layers:
        h = hs[li][0, -1, :].detach().float().cpu().numpy()
        vecs.append(h)
    return np.concatenate(vecs, axis=0)


@torch.no_grad()
def last_layer_hidden_states_sequence(
    bundle: ModelBundle, messages: list[dict[str, str]], assistant_prefill_text: str, layer_index: int = -1
) -> tuple[np.ndarray, list[int]]:
    """
    Returns (H, token_ids) where:
      - token_ids are the tokenized assistant_prefill_text (no special tokens)
      - H is hidden states for those tokens at the specified layer, shape [T, d]
    We compute H by running a forward pass on prompt + prefill, then slicing out the prefill region.
    """
    model, tokenizer = bundle.model, bundle.tokenizer

    base_ids = build_prompt_input_ids(tokenizer, messages, add_generation_prompt=True)
    pre_ids = tokenizer(assistant_prefill_text, add_special_tokens=False, return_tensors="pt").input_ids
    input_ids = torch.cat([base_ids, pre_ids], dim=-1)

    input_ids = input_ids.to(model.device)
    attn = torch.ones_like(input_ids)

    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[layer_index][0]  # [L, d]

    prompt_len = base_ids.shape[-1]
    pre_len = pre_ids.shape[-1]
    H = hs[prompt_len : prompt_len + pre_len, :].detach().float().cpu().numpy()
    token_ids = pre_ids[0].detach().cpu().tolist()
    return H, token_ids


@torch.no_grad()
def conditional_nll_of_prefill(bundle: ModelBundle, messages: list[dict[str, str]], assistant_prefill_text: str) -> float:
    """
    Text-only baseline: mean negative log prob of assistant_prefill tokens conditioned on prompt.
    """
    model, tokenizer = bundle.model, bundle.tokenizer
    base_ids = build_prompt_input_ids(tokenizer, messages, add_generation_prompt=True)
    pre_ids = tokenizer(assistant_prefill_text, add_special_tokens=False, return_tensors="pt").input_ids
    input_ids = torch.cat([base_ids, pre_ids], dim=-1).to(model.device)
    attn = torch.ones_like(input_ids)

    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits  # [1, L, V]

    P = base_ids.shape[-1]
    C = pre_ids.shape[-1]
    if C <= 0:
        return float("nan")

    # Predictions for tokens at positions P..P+C-1 live in logits at positions P-1..P+C-2 (length C).
    pred_logits = logits[:, P - 1 : P + C - 1, :]
    target = input_ids[:, P : P + C]

    log_probs = torch.log_softmax(pred_logits, dim=-1)
    tok_lp = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)  # [1, C]
    nll = (-tok_lp.mean()).item()
    return float(nll)


@torch.no_grad()
def hidden_last_token_and_nll(
    bundle: ModelBundle, messages: list[dict[str, str]], assistant_prefill_text: str, layers: Sequence[int]
) -> tuple[np.ndarray, float]:
    model, tokenizer = bundle.model, bundle.tokenizer

    base_ids = build_prompt_input_ids(tokenizer, messages, add_generation_prompt=True)
    pre_ids = tokenizer(assistant_prefill_text, add_special_tokens=False, return_tensors="pt").input_ids
    input_ids = torch.cat([base_ids, pre_ids], dim=-1).to(model.device)
    attn = torch.ones_like(input_ids)

    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    logits = out.logits  # [1, L, V]

    # hidden at last token for selected layers
    vecs = []
    for li in layers:
        vecs.append(hs[li][0, -1, :].detach().float().cpu().numpy())
    x = np.concatenate(vecs, axis=0)

    # NLL of prefill
    P = base_ids.shape[-1]
    C = pre_ids.shape[-1]
    pred_logits = logits[:, P - 1 : P + C - 1, :]
    target = input_ids[:, P : P + C]
    log_probs = torch.log_softmax(pred_logits, dim=-1)
    tok_lp = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    nll = float((-tok_lp.mean()).item())

    return x, nll
