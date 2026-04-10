# Hermes Memory System Benchmarks

LOCOMO‑style evaluation for the four‑pillar memory upgrade: temporal reasoning, knowledge graph, benchmarking (this), and context paging.

## Test Suite v1.0

**Date**: 2026‑04‑10  
**Environment**: APEX stack (LFM2‑24B, mxbai‑embed‑large‑v1)  
**Store**: LanceDB + BM25 + SQLite for relationships/temporal

### Test Cases

#### 1. Temporal Relative (`temporal_relative`)
- **Goal**: Verify relative date expression parsing (`last week`) filters memories correctly.
- **Memories**:
  - `mem_temp_1`: "I visited Paris last week." (created 3 days ago)
  - `mem_temp_2`: "I will visit Tokyo next month." (created now)
- **Query**: `last week`
- **Expected**: Only `mem_temp_1` returned.

#### 2. Relationship `is_a` (`relationship_is_a`)
- **Goal**: Verify relationship extraction (`is_a`) and `related_to:` filter.
- **Memories**:
  - `mem_rel_1`: "Python is a programming language."
  - `mem_rel_2`: "Python is a snake."
- **Queries**:
  1. `Python` – both memories returned (semantic search).
  2. `related_to:mem_rel_1` – only `mem_rel_2` returned (graph traversal).
- **Metrics**: Precision, recall, F1.

### Running the Benchmarks

```bash
cd /home/lordofwarai/.hermes/hermes-agent
python benchmarks/memory_evaluator.py [--verbose] [--output results.json]
```

By default the evaluator uses a **temporary LanceDB directory**, leaving the production memory store untouched. Use `--production` to run against real data (not recommended).

### Interpreting Scores

- **Precision**: fraction of retrieved memories that are relevant.
- **Recall**: fraction of relevant memories that were retrieved.
- **F1**: harmonic mean of precision and recall.

A perfect system scores 1.0 across all metrics.

### Extending the Test Suite

Add new test cases to `memory_test_set.json` following the schema:

```json
{
  "id": "unique_id",
  "description": "...",
  "memories": [
    {
      "id": "test_mem_1",
      "text": "...",
      "source": "user",
      "memory_type": "fact",
      "epistemic_status": "stated",
      "confidence": 0.8,
      "created_at_offset_days": 0   // optional, relative to now
    }
  ],
  "queries": [
    {
      "query": "...",
      "expected_memory_ids": ["test_mem_1"],
      "mode": "hybrid",
      "top_k": 10
    }
  ]
}
```

### Known Limitations

- **Embedding dependency**: Benchmarks require the local embedding server (mxbai‑embed‑large‑v1 on port 11434) to be running.
- **LLM fallback**: Relationship and temporal extraction may call LFM2‑24B (port 8101). Ensure the model is HOT.
- **Created‑at override**: Temporal offset is not yet implemented; memories use real `created_at`. Future versions will mock timestamps.
- **Isolation**: Temporary store isolation works for LanceDB but not for SQLite relationship/temporal DBs (they still use the profile’s `~/.hermes/` directory). This may cross‑contaminate across runs.

### Future Benchmarks

- **Temporal extraction accuracy**: Compare extracted event dates against ground truth.
- **Graph traversal depth**: Measure recall at different graph depths.
- **Hybrid search fusion**: Test RRF fusion parameters.
- **Context‑paging efficiency**: Token savings with hierarchical memory.