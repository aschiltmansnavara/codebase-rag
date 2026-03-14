# Evaluation Results

**Test set:** 16 questions

## Custom Metrics

| Metric | Score |
|--------|-------|
| avg_keyword_recall | 0.3632 |
| avg_source_precision | 0.1500 |
| questions_answered | 16 |
| questions_failed | 0 |
| avg_latency_s | 6.5990 |

## RAGAS Scores

| Metric | Score |
|--------|-------|
| faithfulness | 0.6667 |
| context_recall | 0.7500 |

## Per-Question Breakdown

| # | Difficulty | Category | Keyword Recall | Docs | Latency | Expected Failure |
|---|-----------|----------|----------------|------|---------|------------------|
| 1 | easy | factual_lookup | 1.00 | 5 | 4.7s | False |
| 2 | easy | factual_lookup | 0.00 | 5 | 3.6s | False |
| 3 | easy | factual_lookup | 0.00 | 5 | 5.0s | False |
| 4 | medium | cross_file_reasoning | 0.43 | 5 | 7.5s | False |
| 5 | medium | factual_lookup | 0.60 | 5 | 6.7s | False |
| 6 | medium | how_does_it_work | 0.50 | 5 | 12.3s | False |
| 7 | medium | cross_file_reasoning | 0.50 | 5 | 3.1s | False |
| 8 | hard | factual_lookup | 0.00 | 5 | 6.7s | False |
| 9 | medium | factual_lookup | 0.00 | 5 | 5.5s | False |
| 10 | medium | factual_lookup | 0.00 | 5 | 7.0s | False |
| 11 | hard | how_does_it_work | 0.20 | 5 | 11.5s | False |
| 12 | hard | cross_file_reasoning | 0.25 | 5 | 7.3s | False |
| 13 | easy | factual_lookup | 1.00 | 5 | 3.6s | False |
| 14 | medium | factual_lookup | 0.00 | 5 | 8.8s | False |
| 15 | hard | cross_file_reasoning | 1.00 | 5 | 9.4s | False |
| 16 | easy | factual_lookup | 0.33 | 5 | 2.7s | True |

## Failure Cases

### Q: What is the minimum Python version required and what build system does the Python package use?

**Expected failure:** Yes

**Answer:** The minimum Python version required is 3.13.3. The Python package uses the Miniforge build system, which is published under the BSD license. It does not have any references to commercially licensed software.
