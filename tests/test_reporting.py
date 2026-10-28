"""Offline checks for readable reports without executing candidate source."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hpc_autooptimizer.agent import OptimizationAgent
from hpc_autooptimizer.cli import main
from hpc_autooptimizer.history import ExperimentHistory
from hpc_autooptimizer.metrics import inference_metrics
from hpc_autooptimizer.models import OptimizationProposal
from hpc_autooptimizer.providers import GenerationResponse, MockLLMProvider, ProviderTimeoutError
from hpc_autooptimizer.reporting import format_cli_summary, format_optimization_summary


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.history = ExperimentHistory(self.root / "experiments")
        self.source = self.root / "source.py"
        self.source.write_text("raise RuntimeError('source must never execute')\n")
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_report_preserves_inference_metrics_and_attributes_model_claims(self):
        proposal = OptimizationProposal.from_json(json.dumps({
            "optimized_code": "raise RuntimeError('candidate must never execute')\n",
            "summary": "This implementation is 10x faster.",
            "changes": ["Reuse intermediate storage"],
            "expected_benefits": ["10x faster and behaviorally equivalent"],
            "risks": ["Floating-point results may differ"], "assumptions": [],
        }))
        body = {"model": "test-model", "response": json.dumps(proposal.to_dict()), "context": list(range(4096)),
                "done": True, "done_reason": "stop", "total_duration": 3_000_000_000,
                "load_duration": 100_000_000, "prompt_eval_duration": 200_000_000,
                "prompt_eval_count": 10, "eval_count": 120, "eval_duration": 2_000_000_000}
        metrics = inference_metrics(body, 3.2)
        response = GenerationResponse("test-model", proposal, metrics=metrics, raw_response=body)
        with patch.object(MockLLMProvider, "generate", return_value=response):
            result = OptimizationAgent(MockLLMProvider(), self.history).optimize(
                self.source.read_text(), "reduce allocations", target="cpu", source_file=self.source)
        record = json.loads((result.artifact_dir / "record.json").read_text())
        summary = (result.artifact_dir / "summary.md").read_text()
        for value in (result.experiment_id, record["timestamp"], str(self.source), "test-model", "reduce allocations"):
            self.assertIn(value, summary)
        for line in ("Target hardware: cpu", "Total inference time: 3.000 s", "Model loading time: 0.100 s",
                     "Prompt processing time: 0.200 s", "Generated tokens: 120", "Generation speed: 60.000 tokens/s",
                     "Python syntax: Valid (syntax only)", "Correctness verification: Not evaluated",
                     "Performance benchmark: Not evaluated", "Model-proposed summary (unverified)",
                     "Expected benefits (model hypotheses, not measurements)", "> This implementation is 10x faster.",
                     "> Floating-point results may differ"):
            self.assertIn(line, summary)
        self.assertIn("> 10x faster and behaviorally equivalent", summary)
        self.assertIn("Behavioral equivalence and performance benefits have not been evaluated.", summary)
        self.assertNotIn(proposal.optimized_code, summary)
        self.assertNotIn('"context"', summary)
        self.assertNotIn("context", record["raw_response"])
        self.assertIn("context", response.raw_response)
        self.assertEqual(record["timing_metrics"], metrics)
        self.assertEqual(record["raw_response"], {key: value for key, value in body.items() if key != "context"})

    def test_mock_summary_missing_metrics_and_source_unchanged(self):
        original = self.source.read_bytes()
        result = OptimizationAgent(MockLLMProvider(), self.history).optimize(
            self.source.read_text(), "speed", source_file=self.source)
        summary = (result.artifact_dir / "summary.md").read_text()
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual((result.artifact_dir / "optimized.py").read_bytes(), original)
        self.assertEqual(json.loads((result.artifact_dir / "proposal.json").read_text()), result.proposal.to_dict())
        self.assertIn("Total inference time: Not available", summary)
        self.assertIn("Generated tokens: Not available", summary)
        self.assertIn("Generation speed: Not available", summary)
        self.assertNotIn("0.000 s", summary)
        self.assertIn("Correctness verification: Not evaluated", summary)
        self.assertIn("Performance benchmark: Not evaluated", summary)

    def test_failed_optimization_also_gets_summary(self):
        with patch.object(MockLLMProvider, "generate", side_effect=ProviderTimeoutError("Inference timed out")):
            with self.assertRaises(ProviderTimeoutError):
                OptimizationAgent(MockLLMProvider(), self.history).optimize("pass", "speed")
        record = self.history.records()[0]
        directory = self.history.root / record["experiment_id"]
        summary = (directory / "summary.md").read_text()
        self.assertIn("No validated proposal was produced.", summary)
        self.assertIn("Python syntax: Not checked", summary)
        self.assertIn("Correctness verification: Not evaluated", summary)
        self.assertIn("> Inference timed out", summary)
        self.assertFalse((directory / "optimized.py").exists())

    def test_older_records_remain_readable_without_rewriting(self):
        directory = self.history.root / ("a" * 32)
        directory.mkdir(parents=True)
        original = '{"experiment_id": "' + "a" * 32 + '", "raw_response": {"context": [1, 2, 3]}}\n'
        record_path = directory / "record.json"
        record_path.write_text(original)
        OptimizationAgent(MockLLMProvider(), self.history).optimize("pass", "speed")
        self.assertEqual(len(self.history.records()), 2)
        self.assertEqual(record_path.read_text(), original)
        self.assertFalse((directory / "summary.md").exists())

    def test_missing_and_zero_optional_metrics(self):
        summary = format_optimization_summary({"timing_metrics": {"total_inference_seconds": 0,
                                               "total_generated_tokens": 0}})
        self.assertIn("Total inference time: 0.000 s", summary)
        self.assertIn("Generated tokens: 0", summary)
        self.assertIn("Generation speed: Not available", summary)
        self.assertIn("Original source file: Not available", summary)
        self.assertIn("Python syntax: Not checked", summary)
        self.assertIn("Total inference time: Not available", format_cli_summary({}))

    def test_default_cli_and_legacy_json_option(self):
        shared = ["--objective", "speed", "--provider", "mock", "--history-dir", str(self.history.root)]
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(["optimize", str(self.source), *shared]), 0)
            text = output.getvalue()
        for label in ("Experiment ID:", "Model: mock-v1", "Output file:", "Total inference time: Not available",
                      "Generation speed: Not available", "Python syntax: Valid (syntax only)",
                      "Correctness verification: Not evaluated", "Performance benchmark: Not evaluated"):
            self.assertIn(label, text)
        self.assertNotIn('"experiment_id"', text)
        self.assertNotIn("raw_response", text)
        self.assertLess(len(text.splitlines()), 12)
        record = self.history.records()[0]
        self.assertIn(record["experiment_id"], text)
        self.assertIn(record["generated_source_path"], text)
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main([str(self.source), *shared, "--json"]), 0)
            result = json.loads(output.getvalue())
        self.assertEqual(result["benchmark_status"], "not_run")
        self.assertEqual(result["model"], "mock-v1")
        self.assertTrue((Path(result["artifact_dir"]) / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
