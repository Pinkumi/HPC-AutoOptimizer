# HPC-AutoOptimizer

HPC-AutoOptimizer is a research project investigating whether large language models can propose useful code optimizations when guided by correctness checks and real hardware feedback. Phase 1 establishes generation, schema validation, and experiment tracking. It does not measure speedups or execute generated programs.

## Architecture

The Python 3.11+ package uses a `src/` layout and has no runtime dependencies beyond the standard library.

| Module | Responsibility |
| --- | --- |
| `providers/base.py` | Abstract LLM interface, structured generation result, typed provider errors |
| `providers/mock.py` | Deterministic offline baseline and injectable response fixtures |
| `providers/ollama.py` | Configurable HTTP(S) inference with socket timeouts |
| `models.py` | Proposal schema and strict JSON validation |
| `config.py` | Validated TOML experiment settings |
| `agent.py` | Objective and source prompt construction, proposal generation, attempt recording |
| `history.py` | JSON records and candidate artifacts stored per experiment |
| `evaluation.py` | Future isolated correctness evaluation contract |
| `benchmark.py` | ParallelBench command planning; candidate execution disabled |
| `cli.py` | Command-line entry point |

The agent sends the objective and source as JSON prompt data, requests a proposal, validates it, and stores it. The source language is unrestricted at this stage: no compilation or language-specific correctness claim is made. The mock provider preserves the supplied source unchanged; it tests the workflow rather than simulating an improvement.

## Installation and offline usage

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
hpc-autooptimizer examples/sum_squares.py --objective "Reduce allocation overhead while preserving behavior" --config examples/mock.toml
```

Alternatively, run without installation:

```bash
PYTHONPATH=src python3 -m hpc_autooptimizer examples/sum_squares.py --objective "Reduce allocation overhead"
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Tests use temporary directories and mocked HTTP transport. They require neither Ollama nor CUDA. Setuptools is needed only to build/install the package; no test framework is required.

## Local Ollama inference

Install Ollama separately, start its service, and download a model suitable for your available memory. The example model is illustrative; a 7B model may exceed a 4 GB GPU's capacity and require CPU offload. This project does not install models, choose quantization, manage GPU placement, or require a GPU.

Edit `examples/ollama.toml` to select an installed model, then run:

```bash
hpc-autooptimizer examples/sum_squares.py --objective "Reduce Python loop overhead, preserving numeric semantics" --config examples/ollama.toml
```

The default local service URL is `http://127.0.0.1:11434`. The provider uses Ollama's [generate API](https://docs.ollama.com/api/generate), requesting non-streaming output with a JSON schema and temperature zero. Model output is still independently validated; temperature zero does not guarantee deterministic real-model inference. HTTP, connection, protocol, and timeout failures have explicit exceptions.

## Remote inference

Copy the configuration to a private, untracked file and replace `base_url` with the HTTPS URL of your inference service, for example `https://inference.example.test`. The same client can reach a Windows-hosted Ollama service through a configured private network such as Tailscale. Routing, HTTPS termination, access control, and server setup are managed separately. HTTPS uses the operating system trust store; certificate verification is not disabled.

No host addresses or credentials are embedded in the implementation. URL credentials are rejected, and this phase has no credential/header authentication support. Prompts contain the entire input source: choose a trusted inference endpoint. The timeout is a socket-operation timeout, not a strict total-generation deadline; there are no automatic retries. For remote inference, increase `timeout_seconds` as needed. TOML-relative `history_dir` paths resolve against the process working directory.

Configuration keys are `provider` (`mock` or `ollama`), `model`, `base_url`, `timeout_seconds`, and `history_dir`. Unknown keys and invalid values fail clearly. Omitting `--config` selects the offline mock defaults. The CLI exits with status 1 for configuration, input, provider, response-validation, or storage failures.

## Structured proposals and history

Every proposal must contain exactly these fields:

- `proposed_source_code`: nonempty string
- `optimization_description`: nonempty string
- `expected_performance_impact`: nonempty string describing a hypothesis
- `assumptions`: array of nonempty strings (may be empty)
- `potential_risks`: array of nonempty strings (may be empty)

Markdown fences, duplicate keys, extra fields, invalid types, and missing fields are rejected. Schema validity establishes only the shape of the response.

Each attempt uses a UUID directory under `history_dir`. `record.json` contains the UTC timestamp, selected model, objective, SHA-256 hashes of UTF-8 source strings, description, generation duration in seconds, schema validation status, correctness status, and benchmark status. Successful proposals also produce `proposal.json` and `proposed_source.txt`. Provider failures and invalid responses are logged with null generated-code fields and an error message; CLI input/configuration errors occur before an attempt begins. Generation duration includes transport and parsing, not hardware execution.

Records are staged in the history directory and published by directory rename. Existing experiment IDs are rejected; there is no mutable shared index. This is a local filesystem store, not a database with crash recovery or distributed transaction guarantees. Disk failures can prevent recording. Artifacts can contain sensitive source and are ignored by Git by default. Original source is hashed rather than retained; preserve the original input separately for reproducibility.

## ParallelBench relationship

[ParallelBench](https://github.com/Pinkumi/ParallelBench) supplies the independent benchmarking engine. Its current README describes built-in matrix multiplication experiments, a `parallelbench` CLI, and `results.json` artifacts. The Phase 1 adapter prepares argument lists for its built-in CPU baseline without launching them. No ParallelBench dependency is installed or repository modified.

The inspected interface does not document an arbitrary generated-source input contract. Baseline timing therefore cannot be attributed to a generated candidate. See [the integration design](docs/parallelbench.md) for the planned boundary.

## Security and current limits

Generated code is untrusted. This package never imports, compiles, or executes candidate programs and uses neither `eval()` nor `exec()`. Correctness and benchmark status remain `not_run`; expected performance text is a model hypothesis, never a measurement. Saving generated text does not make it safe to run manually.

The evaluation interface describes wall-clock, CPU, memory, process, output, network, and filesystem limits. It does not enforce them: evaluation methods fail explicitly. Before enabling execution, implement a separate isolated runner with read-only inputs, a restricted writable directory, no host secrets, disabled networking, enforced limits, process-tree termination, and independent correctness tests. A plain subprocess with a timeout is insufficient. Prompt instructions and JSON validation provide no execution sandbox or proof of semantic equivalence.

## Roadmap

1. Phase 1: provider abstraction, mock/Ollama inference, validated proposals, local history, CLI, and offline tests.
2. Isolated evaluation: threat model, enforced execution policies, reference-based correctness tests, and auditable failure records.
3. ParallelBench integration: agree on a versioned candidate contract, pin a compatible revision, and preserve genuine timing artifacts and hardware metadata.
4. Feedback optimization: compare validated candidates against baselines under controlled conditions, then study iterative search, reproducibility, and model behavior across local and remote inference.
