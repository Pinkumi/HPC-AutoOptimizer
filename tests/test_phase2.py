"""Offline coverage of remote inference, proposal safety, and reproducible records."""
from contextlib import contextmanager
from http.client import IncompleteRead, RemoteDisconnected
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from hpc_autooptimizer.agent import OptimizationAgent, build_prompt, code_hash
from hpc_autooptimizer.cli import main
from hpc_autooptimizer.config import ExperimentConfig
from hpc_autooptimizer.history import ExperimentHistory
from hpc_autooptimizer.llm_benchmark import BENCHMARK_PROMPT, run_llm_benchmark, summarize
from hpc_autooptimizer.metrics import inference_metrics
from hpc_autooptimizer.models import OptimizationProposal, ResponseValidationError
from hpc_autooptimizer.providers import MockLLMProvider, OllamaLLMProvider, ProviderError


PROPOSAL = {"optimized_code": "def square(x):\n    return x * x\n", "summary": "Preserve multiplication",
            "changes": ["Preserve baseline"], "expected_benefits": ["Unverified"],
            "risks": ["Not tested for equivalence"], "assumptions": ["Numeric inputs"]}
MODEL_NAMES = ("qwen3.5:9b-q4_K_M", "qwen2.5-coder:7b")


def envelope(**overrides):
    return {"done": True, "done_reason": "stop", "response": json.dumps(PROPOSAL),
            "total_duration": 3_000_000_000, "load_duration": 100_000_000,
            "prompt_eval_count": 100, "prompt_eval_duration": 1_000_000_000,
            "eval_count": 120, "eval_duration": 2_000_000_000, **overrides}


@contextmanager
def fake_http(responses):
    """Each entry is a JSON body, raw bytes, or exception for one request."""
    entries = iter(responses)

    def open_request(request, **kwargs):
        entry = next(entries)
        if isinstance(entry, BaseException):
            raise entry
        raw = entry if isinstance(entry, bytes) else json.dumps(entry).encode()
        response = io.BytesIO(raw)
        return response

    with patch("hpc_autooptimizer.providers.ollama.urlopen", side_effect=open_request) as mocked:
        yield mocked


class Phase2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.history = ExperimentHistory(self.root / "experiments")
        self.clean_env = patch.dict(os.environ, {}, clear=True)
        self.clean_env.start()
        self.addCleanup(self.clean_env.stop)
        self.provider = OllamaLLMProvider(MODEL_NAMES[0], "https://inference.example.test", host_header="localhost:11434")

    def test_environment_precedence(self):
        toml = self.root / "settings.toml"
        toml.write_text('provider="ollama"\nmodel="from-toml"\nnum_ctx=4096\n')
        dotenv = self.root / ".env"
        dotenv.write_text('HPC_OLLAMA_MODEL="from-file"\nHPC_OLLAMA_HOST_HEADER=localhost:11434\n'
                          'HPC_OLLAMA_THINK=false\nHPC_OLLAMA_KEEP_ALIVE=15m\n')
        config = ExperimentConfig.load(toml, dotenv, environ={"HPC_OLLAMA_MODEL": "from-environment"}, model="from-cli")
        self.assertEqual(config.model, "from-cli")
        self.assertFalse(config.think)
        self.assertEqual(config.num_ctx, 4096)
        self.assertEqual(config.host_header, "localhost:11434")
        self.assertEqual(ExperimentConfig.load(toml, dotenv, environ={}).model, "from-file")
        self.assertIsNone(ExperimentConfig.load(environ={"HPC_OLLAMA_THINK": "null"}).think)

    def test_all_environment_settings(self):
        env = {"HPC_LLM_PROVIDER": "ollama", "HPC_OLLAMA_MODEL": MODEL_NAMES[0],
               "HPC_OLLAMA_BASE_URL": "https://inference.example.test", "HPC_OLLAMA_HOST_HEADER": "localhost:11434",
               "HPC_OLLAMA_TIMEOUT": "190", "HPC_OLLAMA_KEEP_ALIVE": "2m", "HPC_OLLAMA_NUM_CTX": "4096",
               "HPC_OLLAMA_NUM_PREDICT": "512", "HPC_OLLAMA_TEMPERATURE": "0.1", "HPC_OLLAMA_THINK": "true"}
        config = ExperimentConfig.load(environ=env)
        self.assertEqual(config.timeout_seconds, 190)
        self.assertEqual(config.num_predict, 512)
        self.assertTrue(config.think)
        provider = OllamaLLMProvider.from_config(config)
        self.assertEqual(provider.config, config)

    def test_invalid_configuration(self):
        for key, value in (("HPC_OLLAMA_THINK", "yes"), ("HPC_OLLAMA_NUM_CTX", "4096.0"),
                           ("HPC_OLLAMA_TIMEOUT", "nan"), ("HPC_OLLAMA_TEMPERATURE", "-1"),
                           ("HPC_OLLAMA_KEEP_ALIVE", "forever"), ("HPC_OLLAMA_NUM_PREDICT", "-1"),
                           ("HPC_OLLAMA_HOST_HEADER", "localhost\r\nInjected: bad")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ExperimentConfig.load(environ={key: value})
        with self.assertRaisesRegex(ValueError, "Select an Ollama model"):
            ExperimentConfig.load(environ={"HPC_LLM_PROVIDER": "ollama"})
        dotenv = self.root / ".env"
        dotenv.write_text("HPC_OLLAMA_MODEL='unclosed")
        with self.assertRaises(ValueError):
            ExperimentConfig.load(env_file=dotenv, environ={})
        dotenv.write_text("UNKNOWN=bad")
        with self.assertRaises(ValueError):
            ExperimentConfig.load(env_file=dotenv, environ={})

    def test_custom_header_tags_and_generation_settings(self):
        with fake_http([{"models": [{"name": MODEL_NAMES[0]}]}, envelope()]) as http:
            self.provider.require_model()
            self.provider.generate("consistent prompt")
        tags, generate = [call.args[0] for call in http.call_args_list]
        self.assertEqual(tags.method, "GET")
        self.assertTrue(tags.full_url.endswith("/api/tags"))
        for request in (tags, generate):
            self.assertEqual(request.get_header("Host"), "localhost:11434")
        payload = json.loads(generate.data)
        self.assertFalse(payload["think"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["keep_alive"], "15m")
        self.assertEqual(payload["options"], {"temperature": 0.2, "num_ctx": 4096, "num_predict": 2048})
        self.assertEqual(generate.full_url, "https://inference.example.test/api/generate")
        # No unverified SSL context is supplied.
        self.assertEqual(http.call_args.kwargs, {"timeout": 180})

    def test_model_switching_per_request(self):
        with fake_http([envelope(), envelope()]) as http:
            first = self.provider.generate("prompt")
            second = self.provider.generate("prompt", model=MODEL_NAMES[1])
        self.assertEqual([json.loads(call.args[0].data)["model"] for call in http.call_args_list], list(MODEL_NAMES))
        self.assertEqual(first.model, MODEL_NAMES[0])
        self.assertEqual(second.model, MODEL_NAMES[1])
        self.assertEqual(self.provider.model, MODEL_NAMES[0])

    def test_missing_model_and_invalid_tags(self):
        with fake_http([{"models": []}]):
            with self.assertRaisesRegex(ProviderError, "ollama pull"):
                self.provider.require_model()
        with fake_http([{"models": [3]}]):
            with self.assertRaisesRegex(ProviderError, "invalid model list"):
                self.provider.list_models()
        with fake_http([{"models": [{"name": "model:latest"}]}]):
            self.assertEqual(self.provider.require_model("model")["name"], "model:latest")

    def test_unsupported_thinking_fallback_is_explicit(self):
        error = HTTPError("https://inference.example.test", 400, "bad request", {},
                          io.BytesIO(b'{"error":"model does not support thinking"}'))
        with fake_http([error, envelope()]) as http:
            response = self.provider.generate("prompt")
        payloads = [json.loads(call.args[0].data) for call in http.call_args_list]
        self.assertIn("think", payloads[0])
        self.assertNotIn("think", payloads[1])
        self.assertTrue(response.settings["thinking_fallback"])
        self.assertIsNone(response.settings["effective_think"])
        provider = OllamaLLMProvider("test", think=True)
        error = HTTPError("url", 400, "bad", {}, io.BytesIO(b'model does not support thinking'))
        with fake_http([error]) as http:
            with self.assertRaisesRegex(ProviderError, "does not support think=true"):
                provider.generate("prompt")
            self.assertEqual(http.call_count, 1)

    def test_model_default_thinking_omits_parameter(self):
        with fake_http([envelope()]) as http:
            OllamaLLMProvider("test", think=None).generate("prompt")
        self.assertNotIn("think", json.loads(http.call_args.args[0].data))

    def test_reasoning_is_not_candidate_code(self):
        body = envelope(thinking="raise RuntimeError('reasoning must never become code')", context=list(range(4096)))
        with fake_http([body]):
            response = self.provider.generate("prompt")
        self.assertEqual(response.proposal.optimized_code, PROPOSAL["optimized_code"])
        self.assertEqual(response.reasoning, body["thinking"])
        self.assertEqual(response.raw_response, body)

    def test_invalid_generated_python(self):
        for source in ("def broken(:", "```python\npass\n```", "x = ("):
            with self.subTest(source=source), self.assertRaisesRegex(ResponseValidationError, "valid Python"):
                OptimizationProposal.from_json(json.dumps({**PROPOSAL, "optimized_code": source}))

    def test_new_schema_strictness(self):
        for data in ({**PROPOSAL, "changes": "none"}, {**PROPOSAL, "risks": [None]},
                     {**PROPOSAL, "summary": " "}, {**PROPOSAL, "extra": True}):
            with self.subTest(data=data), self.assertRaises(ResponseValidationError):
                OptimizationProposal.from_json(json.dumps(data))
        self.assertEqual(OptimizationProposal.from_json(json.dumps(PROPOSAL)).to_dict(), PROPOSAL)

    def test_incomplete_generation_rejected_and_preserved(self):
        for reason in ("length", "max_tokens", "limit", "unexpected"):
            body = envelope(done_reason=reason)
            with self.subTest(reason=reason), fake_http([body]):
                with self.assertRaises(ResponseValidationError) as failure:
                    self.provider.generate("prompt")
                self.assertEqual(failure.exception.raw_response, body)
                self.assertEqual(failure.exception.metrics["eval_count"], 120)
        with fake_http([envelope(done=False)]):
            with self.assertRaises(ProviderError):
                self.provider.generate("prompt")
        with fake_http([envelope(done_reason={})]):
            with self.assertRaises(ProviderError):
                self.provider.generate("prompt")

    def test_disconnections_and_server_errors(self):
        for failure in (RemoteDisconnected("gone"), IncompleteRead(b"partial"),
                        HTTPError("url", 403, "forbidden", {}, io.BytesIO(b"blocked")),
                        HTTPError("url", 404, "missing", {}, io.BytesIO(b"not found"))):
            with self.subTest(failure=failure), fake_http([failure]):
                with self.assertRaises(ProviderError):
                    self.provider.generate("prompt")
        with fake_http([b'{"bad json"']):
            with self.assertRaises(ProviderError) as failure:
                self.provider.generate("prompt")
            self.assertEqual(failure.exception.raw_response, '{"bad json"')

    def test_metric_conversions(self):
        result = inference_metrics(envelope(), 3.5)
        self.assertEqual(result["total_inference_seconds"], 3)
        self.assertEqual(result["load_seconds"], 0.1)
        self.assertEqual(result["prompt_processing_seconds"], 1)
        self.assertEqual(result["output_generation_seconds"], 2)
        self.assertEqual(result["prompt_tokens_per_second"], 100)
        self.assertEqual(result["output_tokens_per_second"], 60)
        self.assertEqual(result["total_generated_tokens"], 120)
        self.assertEqual(result["client_duration_seconds"], 3.5)

    def test_missing_zero_invalid_metrics(self):
        result = inference_metrics({"eval_count": 0, "eval_duration": 0}, 0)
        self.assertEqual(result["total_generated_tokens"], 0)
        self.assertIsNone(result["output_tokens_per_second"])
        for value in (-1, True, "120", float("nan"), float("inf"), 1.2, 10 ** 1000):
            with self.subTest(value=str(value)[:10]):
                result = inference_metrics({"eval_count": value, "eval_duration": value}, 1)
                self.assertIsNone(result["eval_count"])
                self.assertIsNone(result["output_tokens_per_second"])

    def test_extreme_values_remain_serializable(self):
        result = inference_metrics({"eval_count": 10 ** 308, "eval_duration": 1}, 1)
        self.assertIsNone(result["output_tokens_per_second"])
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            ExperimentConfig(timeout_seconds=10 ** 1000)

    def test_agent_prompt_metadata_and_source_untouched(self):
        source_path = self.root / "input.py"
        source = "def square(x):\n    return x * x\n"
        source_path.write_text(source)
        provider = MockLLMProvider(response=json.dumps(PROPOSAL))
        result = OptimizationAgent(provider, self.history).optimize(
            source, "speed", target="cpu", target_function="square", constraints=["No new dependencies"],
            source_file=source_path, model=MODEL_NAMES[1])
        record = self.history.records()[0]
        prompt = json.loads(record["prompt"])
        self.assertIn("hypotheses", prompt["instruction"])
        self.assertIn("precision", prompt["instruction"])
        self.assertEqual(prompt["target_function"], "square")
        self.assertEqual(prompt["constraints"], ["No new dependencies"])
        self.assertEqual(record["original_source_path"], str(source_path))
        self.assertEqual(record["model_name"], MODEL_NAMES[1])
        self.assertEqual(record["original_code_hash"], code_hash(source))
        self.assertEqual(record["potential_correctness_risks"], PROPOSAL["risks"])
        self.assertEqual(Path(record["generated_source_path"]).read_text(), PROPOSAL["optimized_code"])
        self.assertEqual(source_path.read_text(), source)
        self.assertEqual(result.artifact_dir.parent, self.history.root)

    def test_optimization_persistence_omits_context_and_preserves_metrics(self):
        body = envelope(model=MODEL_NAMES[0], context=list(range(4096)))
        with fake_http([{"models": [{"name": MODEL_NAMES[0]}]}, body]):
            result = OptimizationAgent(self.provider, self.history).optimize("pass", "speed")
        record = self.history.records()[0]
        self.assertEqual(record["raw_response"], {key: value for key, value in body.items() if key != "context"})
        self.assertEqual(record["timing_metrics"], inference_metrics(
            body, record["timing_metrics"]["client_duration_seconds"]))
        self.assertEqual(record["input_token_count"], body["prompt_eval_count"])
        self.assertEqual(record["output_token_count"], body["eval_count"])
        self.assertEqual(result.proposal.to_dict(), PROPOSAL)

    def test_history_compaction_preserves_strings_and_existing_records(self):
        legacy = self.history.root / ("a" * 32) / "record.json"
        legacy.parent.mkdir(parents=True)
        original = json.dumps({"raw_response": {"context": list(range(4096))}})
        legacy.write_text(original)
        metadata = {"experiment_id": "b" * 32, "error": "Server error mentioning context",
                    "raw_response": '{"error": "context unavailable"}', "context": "Research description"}
        directory = self.history.record(metadata)
        self.assertEqual(json.loads((directory / "record.json").read_text()), metadata)
        self.assertEqual(legacy.read_text(), original)

    def test_failure_persistence_preserves_response_except_context(self):
        body = envelope(done_reason="length", context=list(range(4096)))
        with fake_http([{"models": [{"name": MODEL_NAMES[0]}]}, body]):
            with self.assertRaises(ResponseValidationError):
                OptimizationAgent(self.provider, self.history).optimize("pass", "speed")
        record = self.history.records()[0]
        self.assertEqual(record["raw_response"], {key: value for key, value in body.items() if key != "context"})
        self.assertIn("output token limit", record["error"])
        self.assertEqual(record["status"], "failed")
        self.assertIsNone(record["generated_source_path"])
        self.assertEqual(record["correctness_status"], "not_run")
        self.assertFalse((self.history.root / record["experiment_id"] / "optimized.py").exists())

    def test_input_validation_before_inference(self):
        with patch.object(MockLLMProvider, "generate") as generate:
            agent = OptimizationAgent(MockLLMProvider(), self.history)
            for source, settings in (("def broken(:", {}), ("pass", {"target_function": "absent"}),
                                     ("pass", {"target": "tpu"}), ("pass", {"constraints": [1]})):
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    agent.optimize(source, "speed", **settings)
            generate.assert_not_called()

    def test_benchmark_sequential_warmup_stats_and_unload(self):
        bodies = [{"models": [{"name": MODEL_NAMES[0]}]}, envelope(context=list(range(4096))),
                  envelope(context=list(range(4096))), envelope(eval_count=180, context=list(range(4096))),
                  {"done": True}]
        with fake_http(bodies) as http:
            directory, record = run_llm_benchmark(self.provider, self.history, runs=2, warmup=True)
        self.assertTrue((directory / "record.json").is_file())
        requests = [call.args[0] for call in http.call_args_list]
        for request in requests:
            self.assertEqual(request.get_header("Host"), "localhost:11434")
        generation_payloads = [json.loads(request.data) for request in requests[1:-1]]
        self.assertTrue(all(payload["prompt"] == BENCHMARK_PROMPT for payload in generation_payloads))
        self.assertEqual(json.loads(requests[-1].data), {"model": MODEL_NAMES[0], "keep_alive": 0})
        self.assertEqual(record["statistics"]["output_tokens_per_second"],
                         {"mean": 75, "median": 75, "minimum": 60, "maximum": 90, "standard_deviation": 15})
        self.assertEqual(len(record["runs"]), 2)
        self.assertEqual(record["warmup"]["status"], "completed")
        self.assertTrue(record["model_unloaded"])
        self.assertEqual(record["benchmark_status"], "not_run")
        persisted = json.loads((directory / "record.json").read_text())
        self.assertEqual(persisted["statistics"], record["statistics"])
        for saved, original in zip([persisted["warmup"], *persisted["runs"]], [record["warmup"], *record["runs"]]):
            self.assertEqual(saved["raw_response"], {key: value for key, value in original["raw_response"].items()
                                                     if key != "context"})
            self.assertIn("context", original["raw_response"])
            self.assertEqual(saved["metrics"], original["metrics"])

    def test_benchmark_failure_saved_and_model_released(self):
        body = envelope(done_reason="length", context=list(range(4096)))
        with fake_http([{"models": [{"name": MODEL_NAMES[0]}]}, body, {"done": True}]):
            with self.assertRaisesRegex(ProviderError, "Results saved"):
                run_llm_benchmark(self.provider, self.history, runs=3)
        record = self.history.records()[0]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["runs"][0]["status"], "failed")
        self.assertTrue(record["model_unloaded"])
        self.assertIsNone(record["statistics"]["client_duration_seconds"]["mean"])
        compact_response = {key: value for key, value in body.items() if key != "context"}
        self.assertEqual(record["raw_response"], compact_response)
        self.assertEqual(record["runs"][0]["raw_response"], compact_response)

    def test_benchmark_unavailable_and_cleanup_failure(self):
        with fake_http([{"models": []}]):
            with self.assertRaises(ProviderError):
                run_llm_benchmark(self.provider, self.history)
        with fake_http([{"models": [{"name": MODEL_NAMES[0]}]}, envelope(), RemoteDisconnected("gone")]):
            _, record = run_llm_benchmark(self.provider, self.history, runs=1)
        self.assertEqual(record["status"], "completed")
        self.assertFalse(record["model_unloaded"])
        self.assertIn("cleanup_error", record)

    def test_mock_benchmark_does_not_invent_metrics(self):
        _, record = run_llm_benchmark(MockLLMProvider(), self.history, runs=1)
        self.assertIsNone(record["statistics"]["output_tokens_per_second"]["mean"])
        self.assertEqual(summarize([3])["standard_deviation"], 0)
        with self.assertRaises(ValueError):
            run_llm_benchmark(MockLLMProvider(), self.history, runs=0)

    def test_cli_optimize_and_benchmark(self):
        source = self.root / "input.py"
        source.write_text("raise RuntimeError('must never execute')")
        shared = ["--provider", "mock", "--history-dir", str(self.history.root)]
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(main(["optimize", str(source), "--objective", "speed", "--target", "general",
                                   "--model", MODEL_NAMES[1], *shared]), 0)
            result = dict(line.split(": ", 1) for line in out.getvalue().splitlines() if ": " in line)
            self.assertEqual(result["Model"], MODEL_NAMES[1])
            self.assertEqual(Path(result["Output file"]).read_text(), source.read_text())
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(main(["llm-benchmark", "--runs", "2", *shared]), 0)
            self.assertEqual(json.loads(out.getvalue())["status"], "completed")
        source.write_text("def broken(:")
        with patch("sys.stderr", new_callable=io.StringIO), patch.object(MockLLMProvider, "generate") as generate:
            self.assertEqual(main(["optimize", str(source), "--objective", "speed", *shared]), 1)
            generate.assert_not_called()

    def test_cli_ollama_env_and_model_override(self):
        source = self.root / "input.py"
        source.write_text("pass")
        dotenv = self.root / ".env"
        dotenv.write_text(f'HPC_LLM_PROVIDER=ollama\nHPC_OLLAMA_MODEL={MODEL_NAMES[0]}\n'
                          'HPC_OLLAMA_BASE_URL=https://inference.example.test\nHPC_OLLAMA_HOST_HEADER=localhost:11434\n')
        with fake_http([{"models": [{"name": MODEL_NAMES[1]}]}, envelope()]) as http:
            with patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(main(["optimize", str(source), "--objective", "speed", "--env-file", str(dotenv),
                                       "--model", MODEL_NAMES[1], "--history-dir", str(self.history.root)]), 0)
        self.assertEqual(json.loads(http.call_args.args[0].data)["model"], MODEL_NAMES[1])


if __name__ == "__main__":
    unittest.main()
