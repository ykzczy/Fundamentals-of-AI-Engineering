# Week 6 Tutorials: Intelligent Data Analysis Capstone

## Pre-study (Optional Refresher)

Self-learn is optional. If you want extra background:

- [Pre-study guide](../PRESTUDY.md)
- [Self-learn - Chapter 3: AI Engineering Fundamentals](../self_learn/Chapters/3/Chapter3.md)
- [Self-learn - Chapter 4: Hugging Face Platform and Local Inference](../self_learn/Chapters/4/Chapter4.md)

## Overview

Week 6 integrates prior work into one reproducible project:

```text
CSV input -> data overview -> sampled/compressed summary -> LLM interpretation -> report.json + report.md
```

## Main Navigation

- [Simplified project](simplified_project.md)
- [Pipeline design](05_pipeline_design.md)
- [Sampling and compression](06_sampling_compression.md)
- [Capstone requirements](../capstone.md)

## Useful References

- [Tokens and context windows](01_tokens_context.md)
- [Prompt contracts](02_prompt_contracts.md)
- [Structured outputs and validation](03_structured_outputs_validation.md)

## Recommended Order

1. Review the required MVP in `capstone.md`.
2. Build the CSV data overview stage.
3. Add sampling/compression before the LLM call.
4. Add structured LLM interpretation.
5. Write `report.json` and `report.md`.
6. Prepare a short demo and postmortem.
