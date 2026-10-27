import json
import socket
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .base import GenerationResponse, LLMProvider, ProviderError, ProviderTimeoutError
from ..config import ExperimentConfig
from ..models import OptimizationProposal, PROPOSAL_SCHEMA


class OllamaLLMProvider(LLMProvider):
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", timeout_seconds: float = 120):
        super().__init__(model)
        config = ExperimentConfig(provider="ollama", model=model, base_url=base_url,
                                  timeout_seconds=timeout_seconds)
        self.base_url = config.base_url.rstrip("/")
        self.timeout_seconds = config.timeout_seconds

    def generate(self, prompt: str) -> GenerationResponse:
        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "format": PROPOSAL_SCHEMA, "options": {"temperature": 0}}
        request = Request(self.base_url + "/api/generate", data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except (TimeoutError, socket.timeout) as exc:
            raise ProviderTimeoutError("Ollama request timed out") from exc
        except HTTPError as exc:
            raise ProviderError(f"Ollama HTTP error: {exc.code}") from exc
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise ProviderTimeoutError("Ollama request timed out") from exc
            raise ProviderError("Unable to connect to Ollama endpoint") from exc
        except (ValueError, UnicodeError) as exc:
            raise ProviderError("Ollama returned invalid JSON") from exc
        except OSError as exc:
            raise ProviderError("Ollama transport failure") from exc
        if (not isinstance(envelope, dict) or envelope.get("error") or envelope.get("done") is not True
                or not isinstance(envelope.get("response"), str)):
            raise ProviderError("Ollama returned an incomplete or invalid generation envelope")
        return GenerationResponse(self.model, OptimizationProposal.from_json(envelope["response"]))
