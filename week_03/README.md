# Week 3: Data Processing + Reproducible Outputs

Week 3 is the first data-focused practice week. The goal is not to become a Python expert; the goal is to inspect CSV data reproducibly before it is used by models or LLMs.

Python environment setup now happens in [Week 2](../week_02/06_python_environment_setup.md). Week 3 starts with a short preflight check, then focuses on data profiling.

## Pre-study (Optional Refresher)

Self-learn is optional. Use these only if you want extra background:

- [Pre-study guide](../PRESTUDY.md)
- [Week 2 Python environment setup](../week_02/06_python_environment_setup.md)
- [Week 2 AI Python prompt guide](../week_02/ai_python_learning_prompts.md)
- [Self-learn - Chapter 2: Python and Environment Management](../self_learn/Chapters/2/Chapter2.md)

## What You Should Be Able to Do

By the end of this week, you should be able to:

- Activate the Week 2 Python environment and verify `pandas`.
- Load a CSV file with pandas.
- Generate reproducible `profile.json` and `profile.md` outputs.
- Explain at least 3 data quality findings in plain language.

## Python Minimum Checklist

You do not need to become fluent in Python before starting Week 3. You should be able to recognize and ask AI to explain these basics while working through the exercises:

- Variables: store simple values such as file paths, counts, and strings.
- Functions: understand `def load_csv(...):` as a reusable step with inputs and outputs.
- Lists and dictionaries: read simple structures like `["name", "age"]` and `{"rows": 100}`.
- File paths: understand relative paths such as `data/sample.csv` and output paths such as `output/profile.json`.
- Command-line arguments: recognize flags such as `--input data.csv --output_dir output`.
- Error messages: copy the full traceback or terminal output when asking for help.
- Imports: understand that `import pandas as pd` loads a library used by the script.

If any item is unfamiliar, use the linked self-learn chapter as a reference only when needed; it is not a separate prerequisite.

## Week 3 Preflight

Before starting the data profiling script, run:

```bash
python --version
which python
python -c "import pandas as pd; print(pd.__version__)"
```

On Windows, replace `which python` with `where python`.

If pandas is missing, return to [Week 2 Python environment setup](../week_02/06_python_environment_setup.md) and activate/install the environment before continuing.

## Tutorials

Main Week 3 learning path:

- [02_data_profiling_script.md](02_data_profiling_script.md)

Optional reference:

- [01_environment_setup.md](01_environment_setup.md) - archived Week 3 environment lesson; now taught in Week 2

For local inference and LLM reliability topics, see the optional files in [Week 4](../week_04/README.md).

## Workshop Plan

1. Activate the Week 2 environment.
2. Verify pandas imports successfully.
3. Run a small CSV loading example.
4. Build or adapt a data profiling script:
   - input: `--input path/to/data.csv`
   - output directory: `output/`
   - output files: `profile.json` and `profile.md`
5. Run the script on a provided CSV or your own small dataset.
6. Write a short data quality note.

## Deliverables

- A runnable data profiling script.
- `output/profile.json`.
- `output/profile.md`.
- Required profile fields: row/column counts, column names, dtypes, missing values, duplicate rows, numeric summaries, and categorical top values.
- A short report with at least 3 findings.
- A README with setup and run commands.
- Manual test checklist or automated tests.

## Common Pitfalls

- Forgetting to activate the Week 2 environment.
- Running notebooks with the wrong kernel.
- Sharing screenshots of errors without the full text output.
- Writing output files to unclear locations.
- Treating data as "ready" before checking missing values, types, and duplicates.

## Self-check Questions

- Can someone follow your README and reproduce your output files?
- Can you prove you are using the Week 2 virtual environment?
- Can you point to the exact CSV input that produced your report?
- What is the most important data quality issue you found?
