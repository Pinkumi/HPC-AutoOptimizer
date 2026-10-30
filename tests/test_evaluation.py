"""Offline Phase 3 checks. All executed code below is handwritten trusted test data."""
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hpc_autooptimizer.benchmark import CPUFunctionBenchmark, decide_performance, timing_statistics
from hpc_autooptimizer.cli import main
from hpc_autooptimizer.comparison import compare_values, validate_value
from hpc_autooptimizer.evaluation import LocalEvaluator
from hpc_autooptimizer.evaluation_spec import ComparisonSpec, EvaluationSpec, ExecutionLimits
from hpc_autooptimizer.execution import ExecutionBackend, ExecutionResult
from hpc_autooptimizer.history import ExperimentHistory

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/evaluation'


def value(kind, data):
    return {"type": kind, "value": data}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.history = ExperimentHistory(self.root / 'experiments')
        self.backend = ExecutionBackend('trusted', trusted_fixtures=True)
        self.spec = EvaluationSpec.load(ROOT / 'examples/matrix_multiplication.eval.json')

    def test_explicit_contract_and_invalid_settings(self):
        self.assertEqual(self.spec.function, 'matrix_multiplication')
        self.assertEqual(self.spec, EvaluationSpec.from_dict(self.spec.to_dict()))
        for change in ({'unknown': 1}, {'function': 'a.b'}, {'correctness_cases': []},
                       {'benchmark': {'repetitions': 10}}, {'comparison': {'mode': 'guess'}},
                       {'limits': {'timeout_seconds': 0}}, {'limits': {'memory_bytes': True}},
                       {'comparison': {'rtol': -1}}, {'benchmark': {'cases': [{}], 'warmup': -1}},
                       {'benchmark': {'cases': [{}], 'repetitions': 0}},
                       {'benchmark': {'cases': [{}], 'improvement_threshold': 1}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                EvaluationSpec.from_dict({**self.spec.to_dict(), **change})
        path = self.root / 'duplicate.json'
        path.write_text('{"function": "a", "function": "b"}')
        with self.assertRaises(ValueError):
            EvaluationSpec.load(path)

    def test_strict_types_nested_structures_and_integer_precision(self):
        exact = ComparisonSpec()
        self.assertIsNotNone(compare_values(value('int', 1), value('float', 1.0), exact))
        self.assertIsNotNone(compare_values(value('bool', True), value('int', 1), exact))
        self.assertIsNotNone(compare_values(value('list', []), value('tuple', []), exact))
        self.assertIsNotNone(compare_values(value('int', 10 ** 50), value('int', 10 ** 50 + 1), ComparisonSpec('numeric')))
        left = value('list', [value('tuple', [value('int', 3)])])
        right = value('list', [value('tuple', [value('int', 4)])])
        self.assertIn('$[0][0]', compare_values(left, right, exact))
        self.assertIsNone(compare_values(left, left, exact))
        self.assertIsNotNone(compare_values(value('dict', {'a': value('int', 1)}), value('dict', {}), exact))

    def test_numeric_tolerance_nan_infinity_and_explicit_type_opt_in(self):
        spec = ComparisonSpec('numeric', rtol=1e-6, atol=1e-10)
        self.assertIsNone(compare_values(value('float', 1.0), value('float', 1.0000001), spec))
        self.assertIsNotNone(compare_values(value('float', 1.0), value('float', 1.1), spec))
        self.assertIsNotNone(compare_values(value('float', 'nan'), value('float', 'nan'), spec))
        self.assertIsNone(compare_values(value('float', 'nan'), value('float', 'nan'), replace(spec, equal_nan=True)))
        self.assertIsNone(compare_values(value('float', 'inf'), value('float', 'inf'), spec))
        self.assertIsNotNone(compare_values(value('float', 'inf'), value('float', '-inf'), spec))
        self.assertIsNotNone(compare_values(value('int', 1), value('float', 1.0), spec))
        self.assertIsNone(compare_values(value('int', 1), value('float', 1.0), replace(spec, allow_int_float=True)))
        for encoded in ({}, value('float', float('nan')), value('int', True), value('object', 'unsafe')):
            with self.subTest(encoded=encoded), self.assertRaises(ValueError):
                validate_value(encoded)

    def test_correct_matrix_fixture_passes_and_real_samples_are_saved(self):
        path, record = LocalEvaluator(self.backend, self.history).evaluate(
            ROOT / 'examples/matrix_multiplication.py', FIXTURES / 'matmul_correct.py', self.spec)
        self.assertEqual(record['correctness_status'], 'passed')
        self.assertEqual(record['benchmark_status'], 'completed')
        self.assertIn(record['decision'], ('accepted', 'rejected_slower', 'inconclusive'))
        self.assertEqual(len(record['benchmark']['baseline']['samples_ns']), self.spec.benchmark.repetitions)
        self.assertGreater(record['benchmark']['candidate']['mean_ns'], 0)
        self.assertIn('trusted_development_only', (path / 'summary.md').read_text())
        self.assertEqual(json.loads((path / 'record.json').read_text()), record)
        # The familiar [[19, 22], [43, 50]] result is checked without importing the fixture here.
        output = self.backend.run((FIXTURES / 'matmul_correct.py').read_text(), self.spec.function,
                                  [self.spec.correctness_cases[0]], self.spec.limits)
        self.assertEqual([[item['value'] for item in row['value']] for row in output.data['output']['value']],
                         [[19, 22], [43, 50]])

    def test_incorrect_qwen_indexing_is_rejected_before_benchmark(self):
        original = ROOT / 'examples/matrix_multiplication.py'
        candidate = FIXTURES / 'matmul_incorrect.py'
        before = original.read_bytes(), candidate.read_bytes()
        with patch.object(CPUFunctionBenchmark, 'measure') as benchmark:
            _, record = LocalEvaluator(self.backend, self.history).evaluate(original, candidate, self.spec)
            benchmark.assert_not_called()
        self.assertNotEqual(record['correctness_status'], 'passed')
        self.assertEqual(record['benchmark_status'], 'not_run')
        self.assertIsNone(record['benchmark']['speedup'])
        self.assertNotEqual(record['decision'], 'accepted')
        self.assertEqual(record['decision'], 'rejected_incorrect')
        self.assertTrue(record['correctness']['failed_cases'])
        self.assertEqual((original.read_bytes(), candidate.read_bytes()), before)
        for group in ('correctness', 'benchmark'):
            self.assertTrue(any(item['input_group'] == group for item in record['correctness']['cases']))

    def test_fail_closed_without_namespaces_never_executes_files(self):
        backend = ExecutionBackend()
        with patch.object(backend, 'availability', return_value=(False, 'namespaces unavailable')):
            with patch.object(backend, 'run') as run:
                path, record = LocalEvaluator(backend, self.history).evaluate(
                    ROOT / 'examples/matrix_multiplication.py', FIXTURES / 'matmul_correct.py', self.spec)
                run.assert_not_called()
        self.assertEqual(record['correctness_status'], 'isolation_unavailable')
        self.assertEqual(record['decision'], 'evaluation_error')
        self.assertIn('Not evaluated', (path / 'summary.md').read_text())

    def test_invalid_source_and_missing_function_never_run(self):
        original = self.root / 'original.py'
        candidate = self.root / 'candidate.py'
        original.write_text('def matrix_multiplication(a, b):\n    return []\n')
        for source in ('def broken(:', 'def other():\n    return 1\n', 'async def matrix_multiplication(a, b):\n    return []\n'):
            candidate.write_text(source)
            with patch.object(self.backend, 'run') as run:
                _, record = LocalEvaluator(self.backend, self.history).evaluate(original, candidate, self.spec)
                run.assert_not_called()
            self.assertEqual(record['correctness_status'], 'invalid_source')
            self.assertEqual(record['benchmark_status'], 'not_run')

    def test_trusted_mode_requires_both_flags_and_no_fallback(self):
        with self.assertRaises(ValueError):
            ExecutionBackend('trusted')
        with self.assertRaises(ValueError):
            ExecutionBackend('auto', trusted_fixtures=True)
        with patch('hpc_autooptimizer.execution.shutil.which', return_value=None):
            available, reason = ExecutionBackend().availability()
        self.assertFalse(available)
        self.assertIn('no host-execution fallback', reason)

    def test_bubblewrap_command_is_restricted_and_never_mounts_home(self):
        backend = ExecutionBackend('bubblewrap')
        backend.executable = '/usr/bin/bwrap'
        command = backend._sandbox_command(self.root)
        for option in ('--unshare-user', '--unshare-net', '--unshare-pid', '--disable-userns',
                       '--die-with-parent', '--clearenv', '--cap-drop', '--remount-ro', '--ro-bind'):
            self.assertIn(option, command)
        self.assertNotIn('--share-net', command)
        self.assertNotIn(str(Path.home()), command)
        self.assertIn('/work', command)

    def test_worker_timeout_crash_unsupported_and_bounded_output(self):
        fixtures = [
            ('def run():\n    while True: pass\n', 'timeout', 0.3, 4096),
            ('def run():\n    import os\n    os._exit(7)\n', 'execution_error', 2, 4096),
            ('def run():\n    raise ValueError("fixture error")\n', 'execution_error', 2, 4096),
            ('def run():\n    return {1, 2}\n', 'unsupported', 2, 4096),
            ('def run():\n    print("x" * 20000)\n', 'execution_error', 2, 1024),
        ]
        for source, status, timeout, output_bytes in fixtures:
            with self.subTest(status=status):
                result = self.backend.run(source, 'run', [{}], ExecutionLimits(timeout_seconds=timeout, output_bytes=output_bytes))
                self.assertEqual(result.status, status)
                self.assertLessEqual(len(result.stderr), 2000)

    def test_worker_cleans_up_temp_and_does_not_inherit_secrets(self):
        source = 'def run():\n    import os\n    return [os.getcwd(), os.environ.get("HPC_FAKE_SECRET")]\n'
        with patch.dict(os.environ, {'HPC_FAKE_SECRET': 'do-not-copy'}):
            result = self.backend.run(source, 'run', [{}], ExecutionLimits())
        output = result.data['output']['value']
        self.assertFalse(Path(output[0]['value']).exists())
        self.assertEqual(output[1]['value'], None)

    def test_worker_enforces_memory_and_file_limits(self):
        for source, limits in (
            ('def run():\n    return bytearray(128 * 1024 * 1024)\n',
             ExecutionLimits(memory_bytes=64 * 1024 * 1024)),
            ('def run():\n    with open("large.tmp", "wb") as handle:\n        handle.write(b"x" * 4096)\n',
             ExecutionLimits(file_bytes=1024)),
        ):
            with self.subTest(limits=limits):
                result = self.backend.run(source, 'run', [{}], limits)
                self.assertEqual(result.status, 'execution_error')

    def test_reference_failure_prevents_rejection_or_acceptance_claim(self):
        original = self.root / 'original.py'
        original.write_text('def matrix_multiplication(a, b):\n    raise ValueError("bad reference")\n')
        with patch.object(CPUFunctionBenchmark, 'measure') as benchmark:
            _, record = LocalEvaluator(self.backend, self.history).evaluate(
                original, FIXTURES / 'matmul_correct.py', self.spec)
            benchmark.assert_not_called()
        self.assertEqual(record['decision'], 'evaluation_error')
        self.assertEqual(record['correctness_status'], 'execution_error')
        self.assertIn('bad reference', record['correctness']['cases'][0]['reference']['error'])

    def test_input_mutation_is_reset_outside_each_timed_sample(self):
        source = 'def run(items):\n    assert len(items) == 1\n    items.append(2)\n    return len(items)\n'
        result = self.backend.run(source, 'run', [{'args': [[1]]}], ExecutionLimits(),
                                  mode='benchmark', warmup=2, repetitions=4)
        self.assertEqual(result.status, 'ok')
        self.assertEqual(len(result.data['samples_ns']), 4)

    def test_deterministic_decisions_require_multiple_stable_samples(self):
        baseline = timing_statistics([100] * 10)
        for samples, expected in (([90] * 10, 'accepted'), ([110] * 10, 'rejected_slower'),
                                  ([99] * 10, 'inconclusive'), ([10, 90] * 5, 'inconclusive')):
            decision, speedup = decide_performance(baseline, timing_statistics(samples), repetitions=10,
                                                   improvement_threshold=.05, instability_threshold=.10)
            self.assertEqual(decision, expected)
            self.assertIsNotNone(speedup)
        self.assertEqual(decide_performance(baseline, timing_statistics([90]), repetitions=1,
                                            improvement_threshold=.05, instability_threshold=.1)[0], 'inconclusive')

    def test_malformed_or_failed_benchmark_cannot_accept(self):
        for result in (ExecutionResult('ok', {'samples_ns': ['fabricated']}), ExecutionResult('timeout', {}, 'timed out')):
            with patch.object(self.backend, 'run', return_value=result):
                report = CPUFunctionBenchmark(self.backend).measure('unused', 'unused', self.spec, correctness_status='passed')
            self.assertNotEqual(report['status'], 'completed')
            self.assertIsNone(report['speedup'])
        with patch.object(self.backend, 'run') as run:
            with self.assertRaises(ValueError):
                CPUFunctionBenchmark(self.backend).measure('unused', 'unused', self.spec)
            run.assert_not_called()

    def test_cli_evaluation_is_independent_of_ollama_and_leaves_phase2_untouched(self):
        args = ['evaluate', '--original', str(ROOT / 'examples/matrix_multiplication.py'),
                '--candidate', str(FIXTURES / 'matmul_incorrect.py'),
                '--spec', str(ROOT / 'examples/matrix_multiplication.eval.json'), '--history-dir', str(self.history.root)]
        with patch.dict(os.environ, {'HPC_LLM_PROVIDER': 'invalid'}), patch('hpc_autooptimizer.cli.OllamaLLMProvider') as llm:
            with patch('sys.stdout', new_callable=io.StringIO) as stdout:
                code = main([*args, '--backend', 'trusted', '--trusted-fixtures'])
            llm.assert_not_called()
        self.assertIn(code, (1, 2))
        self.assertIn('CPU benchmark: not_run', stdout.getvalue())
        self.assertIn('TRUSTED FIXTURES ONLY', stdout.getvalue())
        with patch('sys.stderr', new_callable=io.StringIO):
            self.assertEqual(main([*args, '--backend', 'trusted']), 1)

    @unittest.skipUnless(os.environ.get('HPC_RUN_SANDBOX_TESTS') == '1', 'optional real namespace test')
    def test_optional_real_sandbox_restricts_home_and_network(self):
        backend = ExecutionBackend()
        available, reason = backend.availability()
        if not available:
            self.skipTest(reason)
        source = ('def run():\n    import os, socket\n'
                  '    try:\n        socket.create_connection(("127.0.0.1", 11434), timeout=.1)\n'
                  '    except OSError:\n        network = False\n'
                  '    else:\n        network = True\n'
                  '    return [os.path.exists("/home"), network]\n')
        result = backend.run(source, 'run', [{}], ExecutionLimits())
        self.assertEqual(result.status, 'ok')
        self.assertEqual([item['value'] for item in result.data['output']['value']], [False, False])


if __name__ == '__main__':
    unittest.main()
