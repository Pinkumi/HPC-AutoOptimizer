# Development guidance

Use Python 3.11+ and keep modules small. Run `PYTHONPATH=src python -m unittest discover -s tests -v` before reporting changes.
Treat generated code as untrusted data: never import or execute it, including during tests. Schema validity does not establish correctness. Keep correctness and benchmark status separate; never invent measurements.
Keep endpoints configurable and credentials out of source control. Do not modify ParallelBench or duplicate its engine. Future execution requires an isolated runner with enforced resource, network, filesystem, and time limits; a plain subprocess is insufficient.
Do not commit or push without an explicit request.
