# Local evaluation (Phase 3 preparation)

This implementation is prepared for offline checking. Phase 3 remains pending end-to-end validation with a working OS sandbox and, later, the remote generation-to-evaluation workflow. No Ollama request is made by `evaluate`; existing inference configuration is untouched. Tests never require Internet access, a GPU, or model downloads.

## Explicit initiation and contract

```bash
hpc-autooptimizer evaluate \
  --original examples/matrix_multiplication.py \
  --candidate /path/to/proposed.py \
  --spec examples/matrix_multiplication.eval.json
```

The original is the behavioral reference, not an independently proven oracle. Both sources are snapshotted and hashed before evaluation; existing source and experiment files are never modified. Syntax and a top-level synchronous function definition are checked with `ast.parse`. Code is imported only by a separate worker after backend selection and isolation checks. `optimize` and `llm-benchmark` never initiate this workflow.

The example JSON specifies:

- `function`: the top-level function to call. Async functions, classes, and automatic function discovery are unsupported.
- `correctness_cases`: nonempty array of `{ "args": [...], "kwargs": {...} }` objects. Inputs are JSON primitives, nested arrays, and string-keyed objects. Test strategy is never inferred from source.
- `comparison`: `mode` (`exact` or `numeric`), finite nonnegative `rtol`/`atol`, `equal_nan` (default false), and `allow_int_float` (default false).
- `benchmark`: explicit nonempty `cases`, `warmup`, `repetitions`, `improvement_threshold` (default 0.05), and `instability_threshold` (coefficient of variation, default 0.10).
- `limits`: per-worker wall timeout, CPU seconds, address-space bytes, process limit, output bytes, and maximum individual file bytes.

Unknown fields, duplicate fields, invalid limits, and non-JSON inputs fail clearly. Benchmark inputs are also correctness-tested before timing. Every invocation receives fresh copied inputs. Only returned values are compared: argument mutation, external side effects, random behavior, and untested inputs are outside this contract.

Outputs retain their types through a bounded JSON protocol: None, bool, int, float, str, list, tuple, and dictionaries with string keys. List/tuple differences and bool/int differences fail. Integers compare exactly even in numeric mode; large integers are never silently converted to float. Int/float comparison requires explicit `allow_int_float: true` in numeric mode. NaNs fail unless `equal_nan` is true; infinities match only the same sign. Arrays and GPU objects are future extensions and currently report `unsupported`.

## Execution boundaries

`auto` supports rootless Bubblewrap on Linux. It checks both the executable and a real namespace probe. It never falls back to host execution. Missing tools, unsupported options, missing `/usr` Python runtimes, denied user namespaces, or sandbox launch failures prevent evaluation; nothing is installed automatically.

The sandbox command requests private user/PID/network/IPC namespaces, dropped capabilities, disabled nested user namespaces, a new session, and death with its parent. It mounts the runtime under `/usr` read-only, compatible system-library symlinks, the temporary worker inputs read-only at `/work`, private `/proc` and minimal `/dev`, and a writable `/tmp` limited to at most 16 MiB. The root, runtime, `/work`, `/proc`, and `/dev` are read-only. The home directory, repository, inference settings, and environment secrets are not mounted or forwarded. The worker enforces no-new-privileges and hard resource limits before importing either evaluated file. `-I -S -B` disables user import paths, site initialization, and bytecode output. This initial worker intentionally uses the standard library only; NumPy/Numba dependency environments are not yet provisioned.

CPU/address-space/process/file limits use Linux `resource` limits. Process limits count the real user's processes and can be stricter than the requested count; address-space limits are per process, not a shared cgroup memory budget. No GPU devices are exposed. A bounded pipe reader terminates workers on excessive combined stdout/stderr or wall timeout. Logs are limited to 2,000 characters per invocation in records. Worker wall duration includes startup/import; function samples exclude it.

A namespace backend is an OS security boundary, but does not prove protection against kernel/runtime vulnerabilities or malicious benchmark forgery. Rootless execution, patched system components, and a dedicated machine are still relevant when evaluating adversarial programs. The current environment rejects namespace creation, so hardened execution has not yet passed live verification. Do not use local fixture results as evidence that this isolation works on the host.

### Trusted fixtures only

For explicitly reviewed, handwritten development fixtures:

```bash
hpc-autooptimizer evaluate \
  --original examples/matrix_multiplication.py \
  --candidate tests/fixtures/evaluation/matmul_correct.py \
  --spec examples/matrix_multiplication.eval.json \
  --backend trusted --trusted-fixtures
```

Replace the candidate with `tests/fixtures/evaluation/matmul_incorrect.py` to reproduce the indexing regression. Square, rectangular, zero-valued, integer, and floating-point inputs are included. The incorrect fixture fails and is never benchmarked.

**Trusted mode is not a sandbox.** It can access host files and networks despite temporary directories, an empty environment, subprocesses, and resource limits. Both flags are required. Never use this mode for LLM output or unknown source. The tests execute only handwritten fixtures and preserve saved Qwen artifacts as inspection data.

## Measurements and decisions

After all explicit correctness and benchmark cases pass, two fresh workers measure the original and candidate under the same limits and input/warm-up/repetition settings. The order is baseline followed by candidate and is recorded; no paired-trial or randomized-order claim is made. Each `perf_counter_ns()` sample measures all benchmark cases once. Input copying, source parsing/import, startup, and warm-up are excluded; Python loop/argument-call overhead is included. Warm-up does not guarantee cache stabilization or JIT completion.

Real nanosecond samples, mean, median, min, max, population standard deviation, and coefficient of variation are saved. Speedup is `baseline_median / candidate_median`. At least three repetitions, nonzero medians, and variability below the configured threshold are required for a performance decision. A candidate with at least 5% median time reduction is `accepted`; at least 5% regression is `rejected_slower`; close or unstable results are `inconclusive`. Thresholds are configurable. This deterministic rule is conservative triage, not a statistical significance test or general performance guarantee. Small example matrices are functional smoke cases, not publication-quality benchmarks.

Correctness mismatch gives `rejected_incorrect`, while invalid source, unsupported output, reference failures, timeouts, unavailable isolation, or measurement errors give `evaluation_error`. Any non-passed correctness status prevents benchmarking. Candidate acceptance never overrides a correctness failure.

## Records and verification

Every evaluation writes a separate UUID directory containing `record.json` and `summary.md`. The record stores source identifiers/hashes, the complete contract, backend/isolation mode, Python/OS metadata, syntax/correctness status, failing case indices and bounded output previews, errors/logs, actual timing samples/statistics, speedup, and decision. No duplicate source files are persisted and prior Phase 2 records/summaries remain unchanged. Phase 2 proposals still say `Not evaluated`; consult the new evaluation summary linked by candidate path/hash.

CLI exit status is 0 for `accepted` or `inconclusive`, 2 for rejection, and 1 for configuration/evaluation errors. Reports always show the actual decision.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
# Optional namespace integration check, still offline and using only a handwritten fixture:
HPC_RUN_SANDBOX_TESTS=1 PYTHONPATH=src python -m unittest discover -s tests -p test_evaluation.py -v
```

The optional test skips if namespaces are unavailable. Remaining work before closing Phase 3: verify the real sandbox on a capable host, review its restrictions, and validate the full workflow when the remote LLM is reachable. ParallelBench remains unchanged; the small local function timer is a bridge for contract validation, not a copy of its numerical benchmark engine. A stable upstream candidate interface and controlled benchmarking methodology remain future integration work.
