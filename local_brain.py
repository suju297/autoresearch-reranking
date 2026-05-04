from __future__ import annotations

import gc
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def _normalize_gpu_ordinal(value: int | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    return int(raw)


DEFAULT_GGUF_MODEL_PATH = str(Path.home() / "Models" / "Qwen" / "Qwen3-8B-Q4_K_M.gguf")
DEFAULT_BRAIN_BACKEND = os.environ.get("AUTORESEARCH_BACKEND", "llama-cpp")
DEFAULT_BRAIN_MODEL = os.environ.get("AUTORESEARCH_MODEL", DEFAULT_GGUF_MODEL_PATH)
DEFAULT_BRAIN_REASONING_MODE = os.environ.get("AUTORESEARCH_REASONING_MODE", "no-think")
DEFAULT_REVIEW_REASONING_MODE = os.environ.get("AUTORESEARCH_REVIEW_REASONING_MODE", "no-think")
DEFAULT_BRAIN_MAX_TOKENS = 3200
DEFAULT_BRAIN_TEMPERATURE = 0.2
DEFAULT_BRAIN_TOP_P = 0.9
DEFAULT_LLAMA_N_CTX = int(os.environ.get("AUTORESEARCH_LLAMA_N_CTX", "8192"))
DEFAULT_LLAMA_N_GPU_LAYERS = int(os.environ.get("AUTORESEARCH_LLAMA_N_GPU_LAYERS", "-1"))
DEFAULT_LLAMA_TYPE_K = _env_optional_int("AUTORESEARCH_LLAMA_TYPE_K")
DEFAULT_LLAMA_TYPE_V = _env_optional_int("AUTORESEARCH_LLAMA_TYPE_V")
DEFAULT_BRAIN_KEEP_LOADED = _env_flag("AUTORESEARCH_KEEP_LOADED", default=False)
DEFAULT_BRAIN_WARM_START = _env_flag("AUTORESEARCH_WARM_START", default=False)
DEFAULT_TRANSFORMERS_DEVICE_MAP = os.environ.get("AUTORESEARCH_TRANSFORMERS_DEVICE_MAP", "auto")
DEFAULT_TRANSFORMERS_DTYPE = os.environ.get("AUTORESEARCH_TRANSFORMERS_DTYPE", "auto")
DEFAULT_SYSTEM_PROMPT = """You are the autonomous research brain for a frozen reranking harness.

Your job is to propose exactly one bounded improvement to rerank_strategy.py.

Rules:
- Edit only rerank_strategy.py.
- Prefer simple changes with a plausible ndcg@10 upside.
- Keep latency risk visible; the harness will reject regressions.
- Return the complete file contents, not a diff.
- Do not return the unchanged file.
- Do not mention any other file or suggest manual steps.

Return exactly this format:
```json
{
  "family": "one allowed search family from the prompt",
  "label": "short-kebab-case-label",
  "summary": "one sentence describing the change",
  "hypothesis": "why this should improve reranking",
  "changed_keys": ["list", "of", "strategy", "keys"],
  "primary_mechanism": "the main mechanism being changed",
  "why_recent_attempts_failed": "why nearby recent attempts failed",
  "why_not_duplicate": "why this proposal is materially different",
  "expected_ndcg_direction": "up|flat|down|slightly_up|slightly_down",
  "expected_recall_direction": "up|flat|down|slightly_up|slightly_down",
  "expected_latency_direction": "up|flat|down|slightly_up|slightly_down",
  "promotion_risk": "low|medium|high"
}
```
```python
<full rerank_strategy.py contents>
```"""

_FAMILY_RE = re.compile(r"^\s*FAMILY:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_LABEL_RE = re.compile(r"^\s*LABEL:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_SUMMARY_RE = re.compile(r"^\s*SUMMARY:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_PYTHON_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_GENERIC_CODE_BLOCK_RE = re.compile(r"```(?!json)(?:[a-zA-Z0-9_-]+)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


@dataclass
class BrainProposal:
    family: str | None
    label: str
    summary: str
    hypothesis: str
    changed_keys: list[str]
    primary_mechanism: str
    why_recent_attempts_failed: str
    why_not_duplicate: str
    expected_ndcg_direction: str
    expected_recall_direction: str
    expected_latency_direction: str
    promotion_risk: str
    strategy_code: str
    raw_response: str
    model_name: str


class ProposalFormatError(ValueError):
    pass


def is_gguf_path(model_name: str) -> bool:
    candidate = Path(model_name).expanduser()
    return candidate.suffix.lower() == ".gguf" or candidate.exists()


def normalize_model_name(model_name: str) -> str:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return str(candidate)
    return model_name


def resolve_transformers_device_map(preferred_gpu: int | None = None):
    preferred = DEFAULT_TRANSFORMERS_DEVICE_MAP.strip().lower()
    if preferred_gpu is not None:
        import torch

        if torch.cuda.is_available():
            return {"": preferred_gpu}
        return "cpu"
    if preferred != "auto":
        return preferred
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "auto"
    return "cpu"


def resolve_transformers_dtype():
    import torch

    preferred = DEFAULT_TRANSFORMERS_DTYPE.strip().lower()
    if preferred == "float16":
        return torch.float16
    if preferred == "bfloat16":
        return torch.bfloat16
    if preferred == "float32":
        return torch.float32
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.float16
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def strip_reasoning(text: str) -> str:
    stripped = _THINK_BLOCK_RE.sub("", text).strip()
    return stripped or text.strip()


def sanitize_label(text: str, fallback: str = "trial") -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not value:
        return fallback
    return re.sub(r"-{2,}", "-", value)


def normalize_changed_keys(values: Sequence[object]) -> list[str]:
    keys = []
    for value in values:
        normalized = re.sub(r"[^a-z0-9_]+", "_", str(value).lower()).strip("_")
        if not normalized:
            continue
        keys.append(normalized)
    deduped: list[str] = []
    for key in keys:
        if key not in deduped:
            deduped.append(key)
    return deduped


def _select_json_block(text: str) -> Mapping[str, Any] | None:
    for block in _JSON_BLOCK_RE.findall(text):
        try:
            loaded = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, Mapping):
            return loaded
    return None


def _select_code_block(text: str) -> str | None:
    matches = [match.strip() for match in _PYTHON_BLOCK_RE.findall(text)]
    if not matches:
        matches = [match.strip() for match in _GENERIC_CODE_BLOCK_RE.findall(text)]
    if not matches:
        return None
    for block in matches:
        if "def rerank(" in block and "def strategy_config(" in block:
            return block
    return matches[0]


def parse_proposal(text: str, *, model_name: str, fallback_label: str) -> BrainProposal:
    cleaned_text = strip_reasoning(text)
    metadata = _select_json_block(cleaned_text) or {}
    code = _select_code_block(cleaned_text)
    if code is None:
        stripped = cleaned_text.strip()
        if "def rerank(" in stripped and "def strategy_config(" in stripped:
            code = stripped
        else:
            raise ProposalFormatError("brain response did not contain a complete python file")

    family_match = _FAMILY_RE.search(cleaned_text)
    label_match = _LABEL_RE.search(cleaned_text)
    summary_match = _SUMMARY_RE.search(cleaned_text)
    family_source = str(metadata.get("family", "")).strip() if metadata else ""
    if not family_source and family_match:
        family_source = family_match.group(1)
    family = sanitize_label(family_source, fallback="") if family_source else None
    if family == "":
        family = None
    label_source = str(metadata.get("label", "")).strip() if metadata else ""
    if not label_source and label_match:
        label_source = label_match.group(1)
    label = sanitize_label(label_source or fallback_label, fallback=fallback_label)
    summary = str(metadata.get("summary", "")).strip() if metadata else ""
    if not summary and summary_match:
        summary = summary_match.group(1).strip()
    summary = summary or "Autonomous strategy proposal."
    strategy_code = code.strip() + "\n"
    return BrainProposal(
        family=family,
        label=label,
        summary=summary,
        hypothesis=str(metadata.get("hypothesis", "")).strip() or summary,
        changed_keys=normalize_changed_keys(metadata.get("changed_keys", [])),
        primary_mechanism=str(metadata.get("primary_mechanism", "")).strip() or summary,
        why_recent_attempts_failed=str(metadata.get("why_recent_attempts_failed", "")).strip(),
        why_not_duplicate=str(metadata.get("why_not_duplicate", "")).strip(),
        expected_ndcg_direction=str(metadata.get("expected_ndcg_direction", "")).strip().lower(),
        expected_recall_direction=str(metadata.get("expected_recall_direction", "")).strip().lower(),
        expected_latency_direction=str(metadata.get("expected_latency_direction", "")).strip().lower(),
        promotion_risk=str(metadata.get("promotion_risk", "")).strip().lower(),
        strategy_code=strategy_code,
        raw_response=text,
        model_name=model_name,
    )


class LlamaCppBrain:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_GGUF_MODEL_PATH,
        max_tokens: int = DEFAULT_BRAIN_MAX_TOKENS,
        temperature: float = DEFAULT_BRAIN_TEMPERATURE,
        top_p: float = DEFAULT_BRAIN_TOP_P,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        n_ctx: int = DEFAULT_LLAMA_N_CTX,
        n_gpu_layers: int = DEFAULT_LLAMA_N_GPU_LAYERS,
        reasoning_mode: str = DEFAULT_BRAIN_REASONING_MODE,
        keep_loaded: bool = DEFAULT_BRAIN_KEEP_LOADED,
        warm_start: bool = DEFAULT_BRAIN_WARM_START,
        type_k: int | None = DEFAULT_LLAMA_TYPE_K,
        type_v: int | None = DEFAULT_LLAMA_TYPE_V,
        preferred_gpu: int | None = None,
    ) -> None:
        self.model_name = normalize_model_name(model_name)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.reasoning_mode = reasoning_mode
        self.keep_loaded = keep_loaded
        self.warm_start = warm_start
        self.type_k = type_k
        self.type_v = type_v
        self.preferred_gpu = _normalize_gpu_ordinal(preferred_gpu)
        self._llm: Any | None = None
        if self.warm_start:
            self.warm()

    def _user_prompt(self, user_prompt: str) -> str:
        if "qwen" in self.model_name.lower() and self.reasoning_mode == "no-think":
            return f"/no_think\n{user_prompt}"
        if "qwen" in self.model_name.lower() and self.reasoning_mode == "think":
            return f"/think\n{user_prompt}"
        return user_prompt

    def ensure_loaded(self) -> Any:
        if self._llm is None:
            from llama_cpp import Llama

            llama_kwargs: dict[str, Any] = {
                "model_path": self.model_name,
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "type_k": self.type_k,
                "type_v": self.type_v,
                "verbose": False,
            }
            if self.preferred_gpu is not None:
                try:
                    import torch

                    if torch.cuda.is_available():
                        device_count = torch.cuda.device_count()
                        if not 0 <= self.preferred_gpu < device_count:
                            raise ValueError(
                                f"preferred GPU {self.preferred_gpu} is out of range for {device_count} visible CUDA devices"
                            )
                        tensor_split = [0.0] * device_count
                        tensor_split[self.preferred_gpu] = 1.0
                        llama_kwargs["main_gpu"] = self.preferred_gpu
                        llama_kwargs["tensor_split"] = tensor_split
                except ImportError:
                    pass

            self._llm = Llama(
                **llama_kwargs,
            )
        return self._llm

    def warm(self) -> None:
        self.ensure_loaded()

    def unload(self, *, force: bool = False) -> None:
        if self.keep_loaded and not force:
            return
        self._llm = None
        gc.collect()

    def complete(self, user_prompt: str) -> str:
        llm = self.ensure_loaded()
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._user_prompt(user_prompt)},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        choice = response["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content") or choice.get("text") or ""
        return str(content)

    def propose(self, user_prompt: str, *, fallback_label: str) -> BrainProposal:
        response = self.complete(user_prompt)
        return parse_proposal(response, model_name=self.model_name, fallback_label=fallback_label)

    def config(self) -> Mapping[str, Any]:
        return {
            "backend": "llama-cpp",
            "model_name": self.model_name,
            "reasoning_mode": self.reasoning_mode,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "type_k": self.type_k,
            "type_v": self.type_v,
            "preferred_gpu": self.preferred_gpu,
            "keep_loaded": self.keep_loaded,
            "warm_start": self.warm_start,
        }


class TransformersBrain:
    def __init__(
        self,
        *,
        model_name: str,
        max_tokens: int = DEFAULT_BRAIN_MAX_TOKENS,
        temperature: float = DEFAULT_BRAIN_TEMPERATURE,
        top_p: float = DEFAULT_BRAIN_TOP_P,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        reasoning_mode: str = DEFAULT_BRAIN_REASONING_MODE,
        keep_loaded: bool = DEFAULT_BRAIN_KEEP_LOADED,
        warm_start: bool = DEFAULT_BRAIN_WARM_START,
        preferred_gpu: int | None = None,
    ) -> None:
        self.model_name = normalize_model_name(model_name)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.reasoning_mode = reasoning_mode
        self.keep_loaded = keep_loaded
        self.warm_start = warm_start
        self.preferred_gpu = _normalize_gpu_ordinal(preferred_gpu)
        self.device_map = resolve_transformers_device_map(self.preferred_gpu)
        self.dtype = resolve_transformers_dtype()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        if self.warm_start:
            self.warm()

    def _user_prompt(self, user_prompt: str) -> str:
        if "qwen" in self.model_name.lower() and self.reasoning_mode == "no-think":
            return f"/no_think\n{user_prompt}"
        if "qwen" in self.model_name.lower() and self.reasoning_mode == "think":
            return f"/think\n{user_prompt}"
        return user_prompt

    def _pretrained_kwargs(self) -> Mapping[str, Any]:
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        if token:
            kwargs["token"] = token
        return kwargs

    def _model_kwargs(self) -> Mapping[str, Any]:
        kwargs = dict(self._pretrained_kwargs())
        kwargs["device_map"] = self.device_map
        kwargs["dtype"] = self.dtype
        return kwargs

    def ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self._pretrained_kwargs())
            if getattr(self._tokenizer, "padding_side", None) is not None:
                self._tokenizer.padding_side = "left"
            if getattr(self._tokenizer, "pad_token_id", None) is None and getattr(self._tokenizer, "eos_token_id", None) is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **self._model_kwargs())
        return self._model, self._tokenizer

    def _input_device(self, model: Any):
        model_device = getattr(model, "device", None)
        if model_device is not None:
            return model_device
        try:
            return next(model.parameters()).device
        except StopIteration:
            import torch

            return torch.device("cpu")

    def warm(self) -> None:
        self.ensure_loaded()

    def unload(self, *, force: bool = False) -> None:
        if self.keep_loaded and not force:
            return
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def complete(self, user_prompt: str) -> str:
        model, tokenizer = self.ensure_loaded()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._user_prompt(user_prompt)},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_device = self._input_device(model)
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0,
        }
        if getattr(tokenizer, "pad_token_id", None) is not None:
            generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
        output = model.generate(**inputs, **generate_kwargs)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def propose(self, user_prompt: str, *, fallback_label: str) -> BrainProposal:
        response = self.complete(user_prompt)
        return parse_proposal(response, model_name=self.model_name, fallback_label=fallback_label)

    def config(self) -> Mapping[str, Any]:
        dtype_name = getattr(self.dtype, "__name__", str(self.dtype))
        return {
            "backend": "transformers",
            "model_name": self.model_name,
            "reasoning_mode": self.reasoning_mode,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "device_map": self.device_map,
            "preferred_gpu": self.preferred_gpu,
            "dtype": dtype_name,
            "keep_loaded": self.keep_loaded,
            "warm_start": self.warm_start,
        }


def build_brain(
    *,
    backend: str = DEFAULT_BRAIN_BACKEND,
    model_name: str = DEFAULT_BRAIN_MODEL,
    max_tokens: int = DEFAULT_BRAIN_MAX_TOKENS,
    temperature: float = DEFAULT_BRAIN_TEMPERATURE,
    top_p: float = DEFAULT_BRAIN_TOP_P,
    llama_n_ctx: int = DEFAULT_LLAMA_N_CTX,
    llama_n_gpu_layers: int = DEFAULT_LLAMA_N_GPU_LAYERS,
    reasoning_mode: str = DEFAULT_BRAIN_REASONING_MODE,
    keep_loaded: bool = DEFAULT_BRAIN_KEEP_LOADED,
    warm_start: bool = DEFAULT_BRAIN_WARM_START,
    llama_type_k: int | None = DEFAULT_LLAMA_TYPE_K,
    llama_type_v: int | None = DEFAULT_LLAMA_TYPE_V,
    preferred_gpu: int | None = None,
):
    if backend == "auto":
        resolved_backend = "llama-cpp" if is_gguf_path(model_name) else "transformers"
    else:
        resolved_backend = backend
    if resolved_backend == "llama-cpp":
        if not is_gguf_path(model_name):
            raise ValueError("llama-cpp brain requires a local GGUF model path")
        return LlamaCppBrain(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n_ctx=llama_n_ctx,
            n_gpu_layers=llama_n_gpu_layers,
            reasoning_mode=reasoning_mode,
            keep_loaded=keep_loaded,
            warm_start=warm_start,
            type_k=llama_type_k,
            type_v=llama_type_v,
            preferred_gpu=preferred_gpu,
        )
    if resolved_backend == "transformers":
        return TransformersBrain(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_mode=reasoning_mode,
            keep_loaded=keep_loaded,
            warm_start=warm_start,
            preferred_gpu=preferred_gpu,
        )
    raise ValueError(f"Unsupported brain backend: {backend}")
