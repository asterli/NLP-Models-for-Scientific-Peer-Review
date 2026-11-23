"""F12: the end-to-end pipeline (paper text -> F06 claims -> F11 retrieval -> F09 comparison -> report).

`core` runs the steps and records what happened at each one. `components` wraps the real F06/F11/F09
callables. `inputs` loads papers. `report` writes the Markdown report. `provenance` records versions.
The command-line entry point is scripts/run_pipeline.py; see docs/pipeline.md.
"""
