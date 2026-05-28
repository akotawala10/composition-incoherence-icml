"""LLM forecasting harness: an LLMClient protocol and concrete clients.

The interface decouples the projection / baseline / metric code from any
specific LLM SDK. A client must produce ``K`` independent probability samples
in ``[0, 1]`` for a single question.

Three clients are provided:
- :class:`MockClient` — deterministic noise around a hidden truth, used for
  unit tests and quick offline experiments.
- :class:`OpenAIClient` — wraps the openai SDK; reads OPENAI_API_KEY.
- :class:`AnthropicClient` — wraps the anthropic SDK; reads ANTHROPIC_API_KEY.

Verbalized-probability prompting follows Tian et al. (2023, arXiv:2305.14975):
elicit a number in [0, 1], then parse with a tolerant regex and fall back to
re-prompting with a constrained format.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..data.paleka import PalekaQuestion
from ..types import Clique

log = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "You are a probabilistic forecaster. Provide your best estimate of the "
    "probability that the following question resolves YES. The probability "
    "must be a single number between 0 and 1.\n\n"
    "Question: {title}\n"
    "Resolution criteria: {body}\n"
    "Resolution date: {resolution_date}\n\n"
    "Respond with ONLY a single number between 0 and 1 (e.g. 0.62). "
    "No words, no percent signs, no commentary."
)

SELF_CONSISTENCY_PROMPT = (
    "You are a probabilistic forecaster. Reason step-by-step about base rates, "
    "reference classes, and how this question relates to logically connected "
    "questions you have seen previously. Then output a probability that is "
    "self-consistent with the laws of probability (e.g. P(Q) + P(neg Q) = 1).\n\n"
    "Question: {title}\n"
    "Resolution criteria: {body}\n"
    "Resolution date: {resolution_date}\n\n"
    "After your reasoning, respond on a final line with ONLY a single number "
    "between 0 and 1 (e.g. 0.62)."
)


JOINT_ELICITATION_PROMPT = (
    "You are a probabilistic forecaster. Below are {m} logically related "
    "forecasting questions. Provide your joint probability assessment, "
    "ensuring your answers are mutually consistent with the laws of "
    "probability (e.g. complementary questions sum to 1; conjunctions are "
    "bounded by their components).\n\n"
    "{enumerated_questions}\n\n"
    "Output {m} probabilities between 0 and 1 as a comma-separated list, in "
    "the SAME ORDER as the questions above. Do not include any other text, "
    "explanation, or labels --- only the comma-separated numbers."
)


def parse_csv_probabilities(text: str, expected_count: int) -> list[float] | None:
    """Parse a comma-separated probability list from free-form LLM output.

    Returns a list of floats in [0, 1] of length `expected_count`, else None.
    Tolerant of trailing prose (takes the first sufficiently-long CSV-like
    fragment), percent signs, and stray spaces.
    """
    if not text:
        return None
    # Find the first chunk that has at least `expected_count - 1` commas.
    candidate_lines = [ln.strip() for ln in text.splitlines() if "," in ln]
    candidates = candidate_lines + [text]
    for line in candidates:
        # Allow `0.62, 0.38, 0.62` or `[0.62, 0.38, 0.62]` etc.
        cleaned = re.sub(r"[\[\](){}<>]", "", line).strip().strip(".")
        parts = [p.strip().strip("'\"") for p in cleaned.split(",")]
        if len(parts) < expected_count:
            continue
        out: list[float] = []
        for p in parts:
            v = parse_verbalized_probability(p)
            if v is None:
                continue
            out.append(v)
            if len(out) == expected_count:
                break
        if len(out) == expected_count:
            return out
    return None


# ---------------------------------------------------------------------------
# Probability parsing  (Tian et al. 2023)
# ---------------------------------------------------------------------------

_PROB_RE = re.compile(
    r"""
    (?P<percent>(?<![.\d])\d{1,3}(?:\.\d+)?)\s*%       # 62%, 62.5%
  | (?P<frac>(?<![.\d])(?:0?\.\d+|1\.0+|1(?!\d)|0(?!\d)))   # 0.62, .62, 1, 0
    """,
    re.VERBOSE,
)


def parse_verbalized_probability(text: str) -> float | None:
    """Return a probability ∈ [0, 1] parsed from free-form LLM output, else None."""
    if not text:
        return None
    for m in _PROB_RE.finditer(text):
        if m.group("percent") is not None:
            try:
                v = float(m.group("percent")) / 100.0
            except ValueError:
                continue
        else:
            try:
                v = float(m.group("frac"))
            except ValueError:
                continue
        if 0.0 <= v <= 1.0:
            return v
    return None


# ---------------------------------------------------------------------------
# LLM client protocol
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    """Minimal interface for a probability-emitting LLM client."""

    name: str

    def forecast_one(
        self, question: PalekaQuestion, *, temperature: float = 0.7, seed: int | None = None
    ) -> float | None:
        """Return a single probability sample for `question`, or None on parse failure."""

    def forecast(
        self,
        question: PalekaQuestion,
        K: int,
        *,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> np.ndarray:
        """Return K i.i.d. probability samples in [0, 1] for `question`.

        Parse failures are dropped; the returned array may have length < K. Callers
        should check ``len(samples) >= 1`` before using.
        """


def forecast_clique(
    client: LLMClient,
    questions: tuple[PalekaQuestion, ...],
    K: int,
    *,
    temperature: float = 0.7,
    seed: int | None = None,
) -> np.ndarray:
    """Return shape (m, K) array of K samples per question in the clique.

    Failures (None) are imputed as 0.5 with a warning so downstream baselines
    can still run; for proper analysis, drop rows where the failure rate is
    high.
    """
    m = len(questions)
    out = np.full((m, K), 0.5, dtype=float)
    for i, q in enumerate(questions):
        samples = client.forecast(q, K, temperature=temperature, seed=seed)
        if len(samples) == 0:
            log.warning("All K samples for question %s failed to parse; using 0.5.", q.id)
            continue
        if len(samples) < K:
            log.info("Only %d/%d samples parsed for question %s; padding with mean.",
                     len(samples), K, q.id)
            mean = float(np.mean(samples))
            padded = np.concatenate([samples, np.full(K - len(samples), mean)])
            out[i] = padded
        else:
            out[i] = samples[:K]
    return out


# ---------------------------------------------------------------------------
# MockClient: deterministic noisy forecaster used in tests
# ---------------------------------------------------------------------------

@dataclass
class MockClient:
    """Deterministic noisy 'forecaster' for tests and offline experiments.

    Returns probability samples concentrated near a hidden ground truth
    derived from the question's `resolution` field, with Gaussian noise.

    Bias modes
    ----------
    `bias` is a global additive shift applied to every sample (every question
    gets the same shift; preserves coherence in expectation).

    `incoherence_std` is the standard deviation of a *per-question* random
    bias drawn deterministically from hash(question.id). This produces a
    forecaster whose population-level marginals are systematically incoherent
    across logically-related questions — i.e. p_F ∉ M_C even as K → ∞ — which
    is what real LLMs do (Paleka et al., 2024). Use this to reproduce the
    paper's non-vanishing-JCD-gain claim in the K-sweep ablation.
    """

    name: str = "mock"
    noise_std: float = 0.15
    bias: float = 0.0
    incoherence_std: float = 0.0
    truth_signal: float = 0.7  # how strongly resolved=True pulls toward 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        # Initialize default RNG; per-call seed overrides.
        self._rng = np.random.default_rng(self.seed)

    def _per_question_bias(self, q: PalekaQuestion) -> float:
        """Deterministic per-question bias (hash-based) for incoherence mode."""
        if self.incoherence_std == 0.0:
            return 0.0
        # Hash the question id stably across runs
        import hashlib
        h = hashlib.sha256(f"{self.seed}|{q.id}".encode()).digest()
        # Map first 8 bytes to a uniform in [0, 1] then to a Gaussian via inverse CDF.
        u = int.from_bytes(h[:8], "big") / float(1 << 64)
        # Avoid u=0 / u=1 numerical issues
        u = min(max(u, 1e-9), 1.0 - 1e-9)
        # Inverse standard normal CDF (Beasley-Springer-Moro approximation via scipy)
        from scipy.special import ndtri
        z = float(ndtri(u))
        return self.incoherence_std * z

    def forecast_one(
        self, question: PalekaQuestion, *, temperature: float = 0.7, seed: int | None = None
    ) -> float | None:
        rng = self._rng if seed is None else np.random.default_rng(seed)
        truth = self._truth_anchor(question)
        scale = self.noise_std * temperature / 0.7
        s = truth + self.bias + self._per_question_bias(question) + rng.normal(0.0, scale)
        return float(np.clip(s, 0.0, 1.0))

    def forecast(
        self,
        question: PalekaQuestion,
        K: int,
        *,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> np.ndarray:
        rng = self._rng if seed is None else np.random.default_rng(seed)
        truth = self._truth_anchor(question)
        scale = self.noise_std * temperature / 0.7
        per_q = self._per_question_bias(question)
        samples = truth + self.bias + per_q + rng.normal(0.0, scale, size=K)
        return np.clip(samples, 0.0, 1.0)

    def _truth_anchor(self, q: PalekaQuestion) -> float:
        if q.resolution is True:
            return 0.5 + self.truth_signal / 2.0   # → 0.85 at signal=0.7
        if q.resolution is False:
            return 0.5 - self.truth_signal / 2.0   # → 0.15 at signal=0.7
        return 0.5  # unresolved → uniform


# ---------------------------------------------------------------------------
# OpenAIClient & AnthropicClient — concrete implementations
# ---------------------------------------------------------------------------

@dataclass
class OpenAICompatibleClient:
    """Client for any OpenAI-API-compatible provider (OpenAI direct, Groq,
    DeepSeek, Together, etc.).

    Provider is selected by setting ``base_url`` and ``api_key_env``. The
    AnthropicClient is separate because the Anthropic SDK has a different
    request shape; the Azure OpenAI wrapper is also separate because Azure
    requires `api_version` + `azure_endpoint` rather than `base_url`.
    """

    model: str = "gpt-4o-mini"
    provider: str = "openai"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    name: str = field(init=False)
    prompt_template: str = DEFAULT_PROMPT
    max_retries: int = 2

    def __post_init__(self) -> None:
        self.name = f"{self.provider}/{self.model}"
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "openai SDK required. Install with: pip install openai>=1.40"
            ) from e
        from openai import OpenAI
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} not set for {self.provider}/{self.model}"
            )
        kwargs: dict = {"api_key": api_key}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    def forecast_one(
        self, question: PalekaQuestion, *, temperature: float = 0.7, seed: int | None = None
    ) -> float | None:
        prompt = self.prompt_template.format(
            title=question.title,
            body=question.body,
            resolution_date=question.resolution_date or "unspecified",
        )
        for attempt in range(self.max_retries + 1):
            try:
                kwargs: dict = dict(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=16,
                )
                if seed is not None and self.provider in ("openai", "azure"):
                    kwargs["seed"] = seed
                resp = self._client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                p = parse_verbalized_probability(text)
                if p is not None:
                    return p
            except Exception as e:  # noqa: BLE001
                log.warning("%s request failed (attempt %d): %s",
                            self.name, attempt + 1, e)
        return None

    def forecast(
        self,
        question: PalekaQuestion,
        K: int,
        *,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> np.ndarray:
        out: list[float] = []
        for k in range(K):
            s = self.forecast_one(
                question, temperature=temperature,
                seed=None if seed is None else seed + k,
            )
            if s is not None:
                out.append(s)
        return np.asarray(out, dtype=float)


# Aliases for clarity / backwards compat
@dataclass
class OpenAIClient(OpenAICompatibleClient):
    provider: str = "openai"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class GroqClient(OpenAICompatibleClient):
    """Groq cloud (OpenAI-compatible). Reads GROQ_API_KEY."""
    model: str = "llama-3.3-70b-versatile"
    provider: str = "groq"
    base_url: str | None = "https://api.groq.com/openai/v1"
    api_key_env: str = "GROQ_API_KEY"


@dataclass
class AzureOpenAIClient:
    """Azure OpenAI client.

    Selects deployment via the env var name in `deployment_env`. Reads
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION.
    """

    deployment_env: str = "AZURE_OPENAI_DEPLOYMENT"
    name: str = field(init=False)
    prompt_template: str = DEFAULT_PROMPT
    max_retries: int = 2
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    endpoint_env: str = "AZURE_OPENAI_ENDPOINT"
    api_version_env: str = "AZURE_OPENAI_API_VERSION"

    def __post_init__(self) -> None:
        try:
            from openai import AzureOpenAI  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("openai>=1.40 required") from e
        from openai import AzureOpenAI
        api_key = os.environ.get(self.api_key_env)
        endpoint = os.environ.get(self.endpoint_env)
        api_version = os.environ.get(self.api_version_env)
        deployment = os.environ.get(self.deployment_env)
        if not all([api_key, endpoint, api_version, deployment]):
            missing = [
                e for e, v in (
                    (self.api_key_env, api_key),
                    (self.endpoint_env, endpoint),
                    (self.api_version_env, api_version),
                    (self.deployment_env, deployment),
                ) if not v
            ]
            raise RuntimeError(f"Missing Azure env vars: {missing}")
        self._deployment = deployment
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.name = f"azure/{deployment}"

    def forecast_one(
        self, question: PalekaQuestion, *, temperature: float = 0.7, seed: int | None = None
    ) -> float | None:
        prompt = self.prompt_template.format(
            title=question.title,
            body=question.body,
            resolution_date=question.resolution_date or "unspecified",
        )
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=16,
                )
                text = resp.choices[0].message.content or ""
                p = parse_verbalized_probability(text)
                if p is not None:
                    return p
            except Exception as e:  # noqa: BLE001
                log.warning("Azure request failed (attempt %d): %s",
                            attempt + 1, e)
        return None

    def forecast(
        self,
        question: PalekaQuestion,
        K: int,
        *,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> np.ndarray:
        out: list[float] = []
        for k in range(K):
            s = self.forecast_one(
                question, temperature=temperature,
                seed=None if seed is None else seed + k,
            )
            if s is not None:
                out.append(s)
        return np.asarray(out, dtype=float)


@dataclass
class AnthropicClient:
    """Wraps anthropic.Anthropic; expects ANTHROPIC_API_KEY in env."""

    model: str = "claude-3-5-sonnet-latest"
    name: str = field(init=False)
    prompt_template: str = DEFAULT_PROMPT
    max_retries: int = 2

    def __post_init__(self) -> None:
        self.name = f"anthropic/{self.model}"
        try:
            from anthropic import Anthropic  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "anthropic SDK required. Install with: pip install anthropic>=0.40"
            ) from e
        from anthropic import Anthropic
        self._client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def forecast_one(
        self, question: PalekaQuestion, *, temperature: float = 0.7, seed: int | None = None
    ) -> float | None:
        # Anthropic API does not support an explicit `seed` parameter; we ignore it.
        del seed
        prompt = self.prompt_template.format(
            title=question.title,
            body=question.body,
            resolution_date=question.resolution_date or "unspecified",
        )
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=16,
                    temperature=temperature,
                )
                text = "".join(
                    block.text for block in resp.content if getattr(block, "text", None)
                )
                p = parse_verbalized_probability(text)
                if p is not None:
                    return p
            except Exception as e:  # noqa: BLE001
                log.warning("Anthropic request failed (attempt %d): %s", attempt + 1, e)
        return None

    def forecast(
        self,
        question: PalekaQuestion,
        K: int,
        *,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> np.ndarray:
        out: list[float] = []
        for _k in range(K):
            s = self.forecast_one(question, temperature=temperature, seed=seed)
            if s is not None:
                out.append(s)
        return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# B3: SelfConsistencyClient — sample-time prompt wrapper
# ---------------------------------------------------------------------------

@dataclass
class SelfConsistencyClient:
    """Wraps any LLMClient and swaps in the self-consistency prompt template.

    Used to implement Baseline B3 in the paper: K samples elicited under a
    'reason about coherence first' prompt instead of the bare default. Uses
    Tian-style verbalized probability extraction at the end.
    """

    inner: LLMClient
    name: str = field(init=False)
    prompt_template: str = SELF_CONSISTENCY_PROMPT

    def __post_init__(self) -> None:
        self.name = f"selfconsistency/{self.inner.name}"
        if hasattr(self.inner, "prompt_template"):
            # Override the inner client's template for the duration of calls
            self._original_template = self.inner.prompt_template
        else:
            self._original_template = None

    def _swap_template(self) -> None:
        if hasattr(self.inner, "prompt_template"):
            self.inner.prompt_template = self.prompt_template

    def _restore_template(self) -> None:
        if self._original_template is not None and hasattr(self.inner, "prompt_template"):
            self.inner.prompt_template = self._original_template

    def forecast_one(
        self, question: PalekaQuestion, *, temperature: float = 0.7, seed: int | None = None
    ) -> float | None:
        self._swap_template()
        try:
            return self.inner.forecast_one(question, temperature=temperature, seed=seed)
        finally:
            self._restore_template()

    def forecast(
        self,
        question: PalekaQuestion,
        K: int,
        *,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> np.ndarray:
        self._swap_template()
        try:
            return self.inner.forecast(question, K, temperature=temperature, seed=seed)
        finally:
            self._restore_template()


# ---------------------------------------------------------------------------
# LocalHFClient — open-weight model via transformers, log-prob over yes/no
# ---------------------------------------------------------------------------

@dataclass
class LocalHFClient:
    """Open-weight LLM client using HuggingFace transformers.

    Computes probability via softmax over the model's logits at the first
    answer token, restricted to the {yes, no} token set. K samples are drawn
    by temperature-sampling the logits at that position. Avoids the verbalized
    parser entirely — reliable for Llama-3.x, Qwen-3, Mistral, gpt-oss, etc.

    Usage:
        client = LocalHFClient(model_id="meta-llama/Llama-3.3-70B-Instruct",
                               device="cuda", dtype="bfloat16")
        samples = client.forecast(question, K=8, temperature=0.7)
    """

    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_input_tokens: int = 2048
    name: str = field(init=False)
    prompt_template: str = DEFAULT_PROMPT
    _model: object = field(default=None, init=False, repr=False)
    _tokenizer: object = field(default=None, init=False, repr=False)
    _yes_ids: tuple[int, ...] = field(default=(), init=False, repr=False)
    _no_ids: tuple[int, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = f"hf/{self.model_id}"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "transformers + torch required. Install with: pip install -e .[hf]"
            ) from e
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, device_map=self.device
        )
        self._model.eval()
        self._yes_ids = self._tokens_for_word(("yes", "Yes", " yes", " Yes", "YES", " YES"))
        self._no_ids = self._tokens_for_word(("no", "No", " no", " No", "NO", " NO"))

    def _tokens_for_word(self, variants: tuple[str, ...]) -> tuple[int, ...]:
        ids: list[int] = []
        for v in variants:
            enc = self._tokenizer(v, add_special_tokens=False)["input_ids"]
            if len(enc) >= 1:
                ids.append(int(enc[0]))
        return tuple(set(ids))

    def _yes_probability(self, prompt: str, *, temperature: float, K: int,
                         seed: int | None) -> np.ndarray:
        import torch
        self._ensure_loaded()
        full_prompt = (
            f"{prompt}\n\n"
            "Answer with a single token: 'yes' or 'no'.\nAnswer:"
        )
        inputs = self._tokenizer(
            full_prompt, return_tensors="pt", truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        with torch.no_grad():
            out = self._model(**inputs)
        logits = out.logits[0, -1]                     # (vocab,)
        yes_logits = logits[list(self._yes_ids)]
        no_logits = logits[list(self._no_ids)]
        # Aggregate over multiple yes/no tokens via logsumexp
        yes_lse = torch.logsumexp(yes_logits, dim=-1)
        no_lse = torch.logsumexp(no_logits, dim=-1)
        log_p_yes_norm = yes_lse - torch.logaddexp(yes_lse, no_lse)
        p_yes = float(torch.exp(log_p_yes_norm).item())

        # K stochastic samples: temperature-perturb the logits and re-evaluate
        if seed is not None:
            torch.manual_seed(seed)
        gumbel_yes = -torch.log(
            -torch.log(torch.rand(K, dtype=torch.float32) + 1e-12) + 1e-12
        )
        gumbel_no = -torch.log(
            -torch.log(torch.rand(K, dtype=torch.float32) + 1e-12) + 1e-12
        )
        scaled_yes = yes_lse.float().cpu() + temperature * gumbel_yes
        scaled_no = no_lse.float().cpu() + temperature * gumbel_no
        # Per-sample renormalized probability of yes
        max_ = torch.maximum(scaled_yes, scaled_no)
        log_p = scaled_yes - max_ - torch.log(
            torch.exp(scaled_yes - max_) + torch.exp(scaled_no - max_)
        )
        samples = torch.exp(log_p).numpy()
        # Anchor the first sample at the deterministic mean for K=1 stability
        if K >= 1:
            samples[0] = p_yes
        return np.clip(samples, 0.0, 1.0)

    def forecast_one(
        self, question: PalekaQuestion, *, temperature: float = 0.7, seed: int | None = None
    ) -> float | None:
        prompt = self.prompt_template.format(
            title=question.title,
            body=question.body,
            resolution_date=question.resolution_date or "unspecified",
        )
        try:
            arr = self._yes_probability(prompt, temperature=temperature, K=1, seed=seed)
            return float(arr[0])
        except Exception as e:  # noqa: BLE001
            log.warning("LocalHF forecast_one failed: %s", e)
            return None

    def forecast(
        self,
        question: PalekaQuestion,
        K: int,
        *,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> np.ndarray:
        prompt = self.prompt_template.format(
            title=question.title,
            body=question.body,
            resolution_date=question.resolution_date or "unspecified",
        )
        try:
            return self._yes_probability(prompt, temperature=temperature, K=K, seed=seed)
        except Exception as e:  # noqa: BLE001
            log.warning("LocalHF forecast failed: %s", e)
            return np.array([])


# Convenience: convert clique-shaped sample matrix into the empirical marginal
# vector p_hat ∈ [0,1]^m used by all baselines + JCD.
def empirical_marginal(samples: np.ndarray) -> np.ndarray:
    """Mean across the K axis. samples shape: (m, K) -> (m,)."""
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D (m, K); got shape {samples.shape}")
    return samples.mean(axis=1)


def _has_clique_attr(c: object) -> bool:
    return isinstance(c, Clique)
