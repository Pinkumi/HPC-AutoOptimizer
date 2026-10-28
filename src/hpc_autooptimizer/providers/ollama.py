"""Ollama HTTP client with verified TLS and optional proxy Host routing."""
from http.client import HTTPException
import json
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import GenerationResponse, LLMProvider, ProviderError, ProviderTimeoutError
from ..config import ExperimentConfig
from ..metrics import inference_metrics
from ..models import OptimizationProposal, PROPOSAL_SCHEMA, ResponseValidationError


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


class OllamaLLMProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", timeout_seconds: float = 180,
                 *, host_header: str | None = None, keep_alive: str = "15m", num_ctx: int = 4096,
                 num_predict: int = 2048, temperature: float = 0.2, think: bool | None = False):
        super().__init__(model)
        self.config = ExperimentConfig(provider="ollama", model=model, base_url=base_url,
                                       timeout_seconds=timeout_seconds, host_header=host_header,
                                       keep_alive=keep_alive, num_ctx=num_ctx, num_predict=num_predict,
                                       temperature=temperature, think=think)
        self.base_url = self.config.base_url.rstrip("/")
        self.timeout_seconds = self.config.timeout_seconds

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "OllamaLLMProvider":
        return cls(config.model, **config.inference_settings())

    def inference_settings(self) -> dict:
        return self.config.inference_settings()

    def _request(self, path: str, payload: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.host_header:
            headers["Host"] = self.config.host_header
        request = Request(self.base_url + path, data=None if payload is None else json.dumps(payload).encode("utf-8"),
                          headers=headers, method="GET" if payload is None else "POST")
        start = perf_counter()
        raw = None
        try:
            # Default urllib HTTPS handling verifies the certificate against the URL hostname,
            # independently of an overridden HTTP Host header.
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
            envelope = json.loads(raw, parse_constant=_reject_json_constant)
        except HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except (OSError, HTTPException):
                body = ""
            finally:
                exc.close()
            message = f"Ollama HTTP error: {exc.code}"
            if exc.code == 404 and payload and "model" in payload:
                message += f". Model {payload['model']!r} may not be installed; check /api/tags and install it on the inference server"
            elif exc.code == 403:
                message += ". Check private-network access and HPC_OLLAMA_HOST_HEADER"
            error = ProviderError(message, raw_response=body,
                                  metrics={"client_duration_seconds": perf_counter() - start})
            error.http_status = exc.code
            raise error from exc
        except TimeoutError as exc:
            raise ProviderTimeoutError(f"Ollama request timed out after a socket timeout of {self.timeout_seconds}s",
                                       metrics={"client_duration_seconds": perf_counter() - start}) from exc
        except URLError as exc:
            message = "Unable to reach Ollama; check the configured URL, Tailscale connection, and server service"
            error_type = ProviderError
            if isinstance(exc.reason, TimeoutError):
                message, error_type = "Ollama request timed out", ProviderTimeoutError
            raise error_type(message, metrics={"client_duration_seconds": perf_counter() - start}) from exc
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ProviderError("Ollama returned invalid JSON", raw_response=raw,
                                metrics={"client_duration_seconds": perf_counter() - start}) from exc
        except (OSError, HTTPException) as exc:
            raise ProviderError("Ollama disconnected or the transport failed; check the inference server",
                                metrics={"client_duration_seconds": perf_counter() - start}) from exc
        if not isinstance(envelope, dict) or envelope.get("error"):
            raise ProviderError("Ollama returned an invalid response or server error", raw_response=envelope,
                                metrics={"client_duration_seconds": perf_counter() - start})
        return {"envelope": envelope, "client_seconds": perf_counter() - start}

    def list_models(self) -> list[dict]:
        envelope = self._request("/api/tags")["envelope"]
        models = envelope.get("models")
        if (not isinstance(models, list) or any(not isinstance(item, dict) or
                not isinstance(item.get("name"), str) for item in models)):
            raise ProviderError("Ollama /api/tags returned an invalid model list", raw_response=envelope)
        return models

    def require_model(self, model: str | None = None) -> dict:
        selected = self.model if model is None else model
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError("model must be a nonempty string")
        canonical = selected if ":" in selected.rsplit("/", 1)[-1] else selected + ":latest"
        for item in self.list_models():
            name = item["name"]
            if name == selected or name == canonical:
                return item
        raise ProviderError(f"Model {selected!r} is not installed on the inference server. "
                            f"Install it there with 'ollama pull {selected}' or select a name from /api/tags")

    def unload(self, model: str | None = None) -> None:
        """Release only the selected model; never unload another user's models."""
        envelope = self._request("/api/generate", {"model": self.model if model is None else model, "keep_alive": 0})["envelope"]
        if envelope.get("done") is not True:
            raise ProviderError("Ollama did not confirm model unload", raw_response=envelope)

    def generate(self, prompt: str, *, model: str | None = None) -> GenerationResponse:
        selected = self.model if model is None else model
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError("model must be a nonempty string")
        payload = {"model": selected, "prompt": prompt, "stream": False, "format": PROPOSAL_SCHEMA,
                   "keep_alive": self.config.keep_alive,
                   "options": {"temperature": self.config.temperature, "num_ctx": self.config.num_ctx,
                               "num_predict": self.config.num_predict}}
        if self.config.think is not None:
            payload["think"] = self.config.think
        settings = self.inference_settings()
        settings.update(effective_think=self.config.think, thinking_fallback=False)
        start = perf_counter()
        try:
            try:
                result = self._request("/api/generate", payload)
            except ProviderError as exc:
                body = str(exc.raw_response).lower()
                unsupported = (getattr(exc, "http_status", None) == 400 and "think" in body
                               and any(word in body for word in ("not support", "unsupported", "does not")))
                if unsupported and self.config.think is True:
                    raise ProviderError("Selected model does not support think=true; use think=false or null",
                                        raw_response=exc.raw_response, metrics=exc.metrics) from exc
                if not unsupported or self.config.think is not False:
                    raise
                # Older/non-thinking models may reject even think:false. Retry once only
                # for that explicit capability error, and disclose the effective settings.
                del payload["think"]
                settings.update(effective_think=None, thinking_fallback=True,
                                thinking_fallback_response=exc.raw_response)
                result = self._request("/api/generate", payload)
            envelope = result["envelope"]
            metrics = inference_metrics(envelope, perf_counter() - start)
            if envelope.get("done") is not True or not isinstance(envelope.get("response"), str):
                raise ProviderError("Ollama returned an incomplete or invalid generation envelope",
                                    raw_response=envelope, metrics=metrics)
            reason = envelope.get("done_reason")
            if reason is not None and not isinstance(reason, str):
                raise ProviderError("Ollama returned an invalid done_reason", raw_response=envelope, metrics=metrics)
            if reason in {"length", "max_tokens", "limit"}:
                raise ResponseValidationError("Generation hit its output token limit; increase num_predict "
                                              "or simplify the request. Candidate was not accepted.",
                                              raw_response=envelope, metrics=metrics)
            if envelope.get("done_reason") not in {None, "stop"}:
                raise ResponseValidationError(f"Unexpected generation stop reason: {envelope.get('done_reason')}",
                                              raw_response=envelope, metrics=metrics)
            try:
                proposal = OptimizationProposal.from_json(envelope["response"])
            except ResponseValidationError as exc:
                exc.raw_response, exc.metrics = envelope, metrics
                raise
        except (ProviderError, ResponseValidationError) as exc:
            exc.settings = settings
            exc.metrics.setdefault("client_duration_seconds", perf_counter() - start)
            raise
        return GenerationResponse(selected, proposal, metrics=metrics, raw_response=envelope,
                                  reasoning=envelope.get("thinking") if isinstance(envelope.get("thinking"), str) else None,
                                  settings=settings)
