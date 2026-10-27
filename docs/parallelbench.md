# ParallelBench integration design

Inspected the public [repository README](https://github.com/Pinkumi/ParallelBench#readme) on 2026-10-26. Its documented boundary includes `--device`, `--sizes`, `--warmup`, `--repetitions`, and `--output`, with schema-versioned JSON results. The adapter uses these options to plan existing baseline workloads only. It returns an argument tuple and expected result path; it does not launch a shell or subprocess.

Future implementation should pin and record a tested ParallelBench revision. A separate isolated worker will own process launch, enforced isolation policy, correctness gates, and artifact transfer. The adapter should reject unsupported result schemas and preserve original results, units, environment, and validation metadata. Failed or missing measurements must remain failures rather than zero timings.

Candidate submission needs an agreed, versioned interface upstream or a stable documented programmatic contract. Do not inject candidate imports into the current engine. Keep candidate correctness validation independent from JSON schema validation, and associate measured results with exact original/candidate hashes and runner settings. Reinspect the upstream contract before implementing execution.
