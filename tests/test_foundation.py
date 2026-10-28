import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from hpc_autooptimizer.agent import OptimizationAgent, code_hash
from hpc_autooptimizer.benchmark import ParallelBenchAdapter
from hpc_autooptimizer.cli import main
from hpc_autooptimizer.config import ExperimentConfig
from hpc_autooptimizer.evaluation import DisabledCorrectnessEvaluator, IsolationPolicy
from hpc_autooptimizer.history import ExperimentHistory
from hpc_autooptimizer.models import OptimizationProposal, PROPOSAL_SCHEMA, ResponseValidationError
from hpc_autooptimizer.providers import MockLLMProvider, OllamaLLMProvider, ProviderError, ProviderTimeoutError


VALID = {"proposed_source_code": "print('candidate')", "optimization_description": "Test proposal",
         "expected_performance_impact": "Unknown", "assumptions": [], "potential_risks": ["Unverified"]}


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.history = ExperimentHistory(self.root / "history")

    def test_mock_deterministic(self):
        provider = MockLLMProvider()
        prompt = json.dumps({"source_code": "raise RuntimeError('must never run')"})
        self.assertEqual(provider.generate(prompt), provider.generate(prompt))
        self.assertEqual(provider.generate(prompt).proposal.proposed_source_code, json.loads(prompt)["source_code"])

    def test_schema_rejections(self):
        bad = ["not json", "[]", "{}", json.dumps({**VALID, "extra": 1}),
               json.dumps({**VALID, "assumptions": "none"}),
               json.dumps({**VALID, "potential_risks": [3]}),
               json.dumps({**VALID, "proposed_source_code": " "}),
               '{"a": 1, "a": 2}']
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ResponseValidationError):
                OptimizationProposal.from_json(raw)

    def test_ollama_request(self):
        with patch("hpc_autooptimizer.providers.ollama.urlopen") as open_mock:
            open_mock.return_value.__enter__.return_value.read.return_value = json.dumps(
                {"done": True, "response": json.dumps(VALID)}).encode()
            result = OllamaLLMProvider("test-model", "https://inference.example.test/ollama/", 9).generate("prompt")
            request = open_mock.call_args.args[0]
            self.assertEqual(request.full_url, "https://inference.example.test/ollama/api/generate")
            self.assertEqual(request.method, "POST")
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], "test-model")
            self.assertEqual(payload["prompt"], "prompt")
            self.assertEqual(payload["format"], PROPOSAL_SCHEMA)
            self.assertFalse(payload["stream"])
            self.assertEqual(open_mock.call_args.kwargs["timeout"], 9)
            self.assertEqual(result.proposal.proposed_source_code, VALID["proposed_source_code"])

    def test_timeouts(self):
        for error in (TimeoutError(), URLError(TimeoutError())):
            with self.subTest(error=error), patch("hpc_autooptimizer.providers.ollama.urlopen", side_effect=error):
                with self.assertRaises(ProviderTimeoutError):
                    OllamaLLMProvider("test").generate("prompt")

    def test_transport_errors(self):
        for error in (URLError("offline"), HTTPError("url", 500, "failure", {}, None), OSError("reset")):
            with self.subTest(error=error), patch("hpc_autooptimizer.providers.ollama.urlopen", side_effect=error):
                with self.assertRaises(ProviderError):
                    OllamaLLMProvider("test").generate("prompt")

    def test_bad_envelopes(self):
        for raw in (b"bad", b"[]", b'{}', b'{"done": false, "response": "{}"}',
                    b'{"done": true, "response": 1}', b'{"error": "failed"}'):
            with self.subTest(raw=raw), patch("hpc_autooptimizer.providers.ollama.urlopen") as transport:
                transport.return_value.__enter__.return_value.read.return_value = raw
                with self.assertRaises(ProviderError):
                    OllamaLLMProvider("test").generate("prompt")

    def test_config(self):
        path = self.root / "config.toml"
        path.write_text('provider = "ollama"\nmodel = "custom"\nbase_url = "https://llm.example.test"\n')
        self.assertEqual(ExperimentConfig.from_toml(path).model, "custom")
        for settings in ({"provider": "bad"}, {"timeout_seconds": 0}, {"timeout_seconds": True},
                         {"timeout_seconds": float("nan")}, {"model": ""},
                         {"base_url": "file:///tmp/test"}, {"base_url": "https://u:p@example.test"},
                         {"base_url": "http://example.test:bad"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ExperimentConfig(**settings)
        path.write_text('unknown = 1')
        with self.assertRaises(ValueError):
            ExperimentConfig.from_toml(path)

    def test_agent_history_and_no_execution(self):
        source = "raise RuntimeError('not executed')"
        result = OptimizationAgent(MockLLMProvider(), self.history).optimize(source, "reduce allocations")
        record = self.history.records()[0]
        self.assertEqual(record["original_code_hash"], code_hash(source))
        self.assertEqual(record["generated_code_hash"], code_hash(source))
        self.assertEqual(record["validation_status"], "schema_valid")
        self.assertEqual(record["correctness_status"], "not_run")
        self.assertEqual(record["benchmark_status"], "not_run")
        self.assertGreaterEqual(record["generation_duration_seconds"], 0)
        self.assertEqual((result.artifact_dir / "proposed_source.txt").read_text(), source)
        self.assertEqual(json.loads((result.artifact_dir / "proposal.json").read_text()), result.proposal.to_dict())
        with self.assertRaises(FileExistsError):
            self.history.record(record)
        self.assertEqual(len(self.history.records()), 1)

    def test_invalid_response_logged(self):
        with self.assertRaises(ResponseValidationError):
            OptimizationAgent(MockLLMProvider(response="{}"), self.history).optimize("pass", "speed")
        record = self.history.records()[0]
        self.assertEqual(record["validation_status"], "invalid_response")
        self.assertIsNone(record["generated_code_hash"])
        self.assertFalse((self.history.root / record["experiment_id"] / "proposal.json").exists())

    def test_provider_failure_logged(self):
        with patch.object(MockLLMProvider, "generate", side_effect=ProviderTimeoutError("timeout")):
            with self.assertRaises(ProviderTimeoutError):
                OptimizationAgent(MockLLMProvider(), self.history).optimize("pass", "speed")
        self.assertEqual(self.history.records()[0]["validation_status"], "provider_error")

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            OptimizationAgent(MockLLMProvider(), self.history).optimize("", "speed")
        self.assertEqual(self.history.records(), [])

    def test_cli(self):
        source = self.root / "source.py"
        source.write_text("pass")
        config = self.root / "config.toml"
        config.write_text(f'history_dir = {json.dumps(str(self.history.root))}')
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main([str(source), "--objective", "speed", "--config", str(config)]), 0)
            self.assertIn("Performance benchmark: Not evaluated", output.getvalue())
        with patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(main([str(self.root / "missing"), "--objective", "speed"]), 1)

    def test_execution_interfaces_disabled(self):
        policy = IsolationPolicy(10, 1000000, 5, 1, 10000, self.root)
        with self.assertRaises(NotImplementedError):
            DisabledCorrectnessEvaluator().evaluate(self.root, self.root, policy)
        adapter = ParallelBenchAdapter()
        plan = adapter.plan_baseline(self.root)
        self.assertEqual(plan.argv[:3], ("parallelbench", "--device", "cpu"))
        self.assertFalse(plan.candidate_supported)
        with self.assertRaises(NotImplementedError):
            adapter.evaluate_candidate(self.root, policy)
        with self.assertRaises(ValueError):
            adapter.plan_baseline(self.root, sizes=(0,))


if __name__ == "__main__":
    unittest.main()
