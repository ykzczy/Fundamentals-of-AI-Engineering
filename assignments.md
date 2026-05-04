# Fundamentals of AI Engineering: Assignments & Assessment

## Overview

This 6-week beginner course assesses steady progress from AI tool fluency to a small end-to-end AI engineering project. The grading focus is:

- Can the student run the work?
- Can the student explain the work?
- Are inputs, outputs, and AI assistance documented?
- Did the student verify the result instead of blindly trusting AI?

| Component | Weight | Week Due |
|-----------|--------|----------|
| Week 1: Agent Tools Experience Report | 10% | Week 1 |
| Week 2: AI-Assisted Code Practice | 10% | Week 2 |
| Week 3: Data Profiling Report | 15% | Week 3 |
| Week 4: LLM Structured Output + Reliability Practice | 15% | Week 4 |
| Week 5: ML Baseline Comparison | 10% | Week 5 |
| Week 6: Intelligent Data Analysis Capstone | 25% | Week 6 |
| Participation | 15% | Ongoing |
| **Total** | **100%** | |

## General Submission Guidelines

Weeks 1-2 may be submitted as reports plus screenshots/prompts. Weeks 3-6 should include runnable files when code is required.

Recommended structure for technical weeks:

```text
submission-weekN/
├── README.md
├── requirements.txt
├── src/ or scripts/
├── output/
├── report.md
└── prompts.md or ai_usage.md
```

### README Requirements

Your README should include:

- Environment setup: Python version and dependencies.
- How to run: exact commands.
- Expected outputs: filenames and examples.
- Failure notes: at least one issue you saw or tested.

### AI Tool Usage Declaration

AI tools are encouraged, but transparency is required. Each assignment should include:

- Tools used.
- Representative prompts.
- Which parts were AI-assisted.
- What you personally verified or changed.

Example:

```markdown
## AI Tool Declaration

- Tool used: ChatGPT and Cursor
- Key prompts:
  - "Explain this function line by line."
  - "Help me debug this pandas FileNotFoundError."
- AI-assisted sections:
  - First draft of README troubleshooting notes
  - Suggested fix for one parsing bug
- My contribution:
  - Tested the command, checked the output files, and rewrote the explanation in my own words
```

## Week 1: Agent Tools Experience Report (10%)

### Goal

Explore AI tools and learn how to ask, refine, verify, and reflect. No code submission is required.

### Requirements

1. Try at least 2 AI tools, such as ChatGPT, Claude, Cursor, Kilo, or Copilot Chat.
2. Complete 3 tasks, for example:
   - Ask an AI to explain a concept.
   - Ask an AI to summarize or rewrite content.
   - Ask an AI editor/agent to explain a project folder or simple file.
3. Document what worked, what failed, and what you learned about prompting.

### Deliverables

| File | Description |
|------|-------------|
| `report.md` | 800-1000 word reflection |
| `prompts.md` | Representative prompts and short notes |
| `output/` | Screenshots, copied outputs, or interaction notes |
| `README.md` | Brief overview of what you submitted |

### Assessment Criteria

| Criterion | Weight |
|-----------|--------|
| Completion of 3 meaningful tasks | 30% |
| Prompt documentation | 20% |
| Reflection on strengths and limits | 30% |
| Organization and clarity | 15% |
| AI declaration | 5% |

## Week 2: AI-Assisted Code Practice (10%)

### Goal

Use AI to read, modify, and debug simple Python code from `week_02/code_templates/`. This is guided practice, not a full programming project.

### Requirements

1. Use the provided templates in `week_02/code_templates/`.
2. Ask AI to explain at least 5 functions or code blocks.
3. Make 2-3 small modifications, such as changing a message, adding an input option, or adjusting a data transformation.
4. Complete one debugging record using `debugging_practice.py` or another provided exercise.

### Deliverables

| File | Description |
|------|-------------|
| `report.md` | Code reading, modification, and debugging reflection |
| `modified_code/` | Copies of modified template files |
| `debugging_record.md` | Error, prompt, fix, and verification |
| `prompts.md` | Key prompts used |
| `README.md` | What to run and what changed |

### Assessment Criteria

| Criterion | Weight |
|-----------|--------|
| Code explanation quality | 25% |
| Small modifications completed | 25% |
| Debugging record and verification | 25% |
| Reflection on AI-assisted workflow | 15% |
| AI declaration | 10% |

## Week 3: Data Profiling Report (15%)

### Goal

Create a reproducible data profiling workflow that reads a CSV and produces JSON and Markdown outputs.

### Requirements

1. Create or use a clean Python environment.
2. Load a CSV dataset with pandas.
3. Generate a profiling report that includes:
   - Row and column counts.
   - Column names and inferred types.
   - Missing value counts.
   - Duplicate row count.
   - Basic numeric statistics where applicable.
   - Basic categorical value counts where applicable.
4. Output both `profile.json` and `profile.md`.
5. Include a manual test checklist or automated tests.

### Deliverables

| File | Description |
|------|-------------|
| `src/profiler.py` or `data_profile.py` | Data profiling script |
| `output/profile.json` | Machine-readable profile |
| `output/profile.md` | Human-readable profile |
| `report.md` | Short data quality analysis with at least 3 findings |
| `README.md` | Setup and run instructions |
| `requirements.txt` | Dependencies |

### Assessment Criteria

| Criterion | Weight |
|-----------|--------|
| Script runs and produces expected outputs | 30% |
| Data profiling completeness | 25% |
| Reproducibility and README clarity | 20% |
| Data quality interpretation | 15% |
| Test checklist or tests | 5% |
| AI declaration | 5% |

## Week 4: LLM Structured Output + Reliability Practice (15%)

### Goal

Build a small LLM workflow that produces structured output and handles common API/local inference failures in a beginner-friendly way.

### Requirements

1. Explain tokens, context windows, and prompt contracts in your own words.
2. Design a structured prompt with an explicit JSON output shape.
3. Run the prompt on at least 3 test inputs.
4. Parse and validate the output.
5. Add basic reliability practices:
   - Timeout or max wait setting.
   - Retry limit or repair attempt.
   - Clear error message.
   - Simple logging or saved raw responses.
6. Use either a hosted API or Ollama/local inference. Local-vs-cloud comparison is optional.

### Deliverables

| File | Description |
|------|-------------|
| `src/` or `scripts/` | Structured-output demo and/or simplified LLM client |
| `output/` | Test inputs, raw responses, parsed JSON outputs |
| `report.md` | Explanation of prompt design and one reliability failure mode |
| `README.md` | Setup and run instructions |
| `prompts.md` | Prompt contract and repair prompt if used |

### Assessment Criteria

| Criterion | Weight |
|-----------|--------|
| Structured prompt design | 25% |
| Parseable/validated outputs | 25% |
| Basic reliability handling | 20% |
| Clear failure-mode reflection | 15% |
| Reproducibility and documentation | 10% |
| AI declaration | 5% |

## Week 5: ML Baseline Comparison (10%)

### Goal

Understand the ML training loop by running two lightweight baseline experiments and comparing the results.

### Requirements

1. Use the provided Week 5 examples or a simple tabular dataset.
2. Run two baseline experiments:
   - Change one model type, parameter, seed, or feature choice.
   - Keep the comparison simple and explainable.
3. Save or record the config and metrics for each run.
4. Write a short comparison: what changed, what happened, and what you would try next.

Cross-validation, statistical significance, complex visualizations, and multi-model systems are not required.

### Deliverables

| File | Description |
|------|-------------|
| `train.py` or notebook/script | Training workflow |
| `output/` | Metrics, configs, or comparison table |
| `report.md` | Short experiment comparison |
| `README.md` | Reproduction instructions |
| `requirements.txt` or `pyproject.toml` | Dependencies |

### Assessment Criteria

| Criterion | Weight |
|-----------|--------|
| Two runs completed | 25% |
| Metrics/configs saved or clearly recorded | 25% |
| Comparison explanation | 25% |
| Reproducibility | 15% |
| AI declaration | 10% |

## Week 6: Intelligent Data Analysis Capstone (25%)

### Goal

Build a small reproducible project that reads CSV data and produces an AI-assisted analysis report.

Required MVP:

```text
CSV input -> data overview/statistics -> sampled/compressed summary -> LLM interpretation -> report.json + report.md
```

### Requirements

1. Accept a CSV file path as input.
2. Generate traditional data statistics:
   - Column types.
   - Missing values.
   - Duplicate rows.
   - Basic numeric/categorical summaries.
   - Simple anomaly hints.
3. Avoid sending the full dataset to the LLM; use sampling or compression.
4. Ask an LLM for insights, recommendations, and risk notes using a structured prompt.
5. Write:
   - `report.json` with a stable schema.
   - `report.md` for human readers.
6. Include a README, dependencies, sample output, and one postmortem/reflection.

### Deliverables

| File | Description |
|------|-------------|
| Source code | CLI script or small modular project |
| `output/report.json` | Machine-readable final report |
| `output/report.md` | Human-readable final report |
| `README.md` | One-command run instructions |
| `requirements.txt` or `pyproject.toml` | Dependencies |
| `postmortem.md` | One issue encountered and how it was handled |
| `prompts.md` or `ai_usage.md` | Prompt and AI usage notes |

### Assessment Criteria

| Criterion | Weight |
|-----------|--------|
| End-to-end functionality | 30% |
| Data profiling and sampling quality | 20% |
| LLM interpretation and structured output | 20% |
| Reproducibility and documentation | 15% |
| Reflection/postmortem | 10% |
| AI declaration | 5% |

### Stretch Goals

These are optional and should not replace the MVP:

- Add charts to the Markdown report.
- Support Excel input.
- Support both hosted API and Ollama backends.
- Add caching based on input file hash.
- Add a CLI flag for different report styles.

## Participation (15%)

Participation includes:

| Component | Weight |
|-----------|--------|
| In-class activities | 6% |
| Discussion and questions | 4% |
| Peer feedback | 3% |
| Progress check-ins | 2% |

## Summary Timeline

| Week | Assignment | Due | Weight |
|------|------------|-----|--------|
| 1 | Agent Tools Experience Report | Sunday Week 1 | 10% |
| 2 | AI-Assisted Code Practice | Sunday Week 2 | 10% |
| 3 | Data Profiling Report | Sunday Week 3 | 15% |
| 4 | LLM Structured Output + Reliability Practice | Sunday Week 4 | 15% |
| 5 | ML Baseline Comparison | Sunday Week 5 | 10% |
| 6 | Intelligent Data Analysis Capstone | Sunday Week 6 | 25% |
| 1-6 | Participation | Ongoing | 15% |
