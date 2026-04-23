# Synthetic Financial Dataset

Synthetic data for testing and demoing the RAG pipeline. All companies, names, and figures are fictional.

## Source Documents (`raw/`)

| File | Type | Period |
|------|------|--------|
| `novatech_q1_2024.txt` | Quarterly earnings report | Q1 2024 |
| `novatech_q2_2024.txt` | Quarterly earnings report | Q2 2024 |
| `meridian_capital_q1_2024.txt` | LP investor letter | Q1 2024 |
| `apex_holdings_annual_2023.txt` | Annual report | FY2023 |
| `cornerstone_bank_q3_2024.txt` | Quarterly earnings report | Q3 2024 |
| `market_outlook_q3_2024.txt` | Analyst market outlook | Q3 2024 |
| `greenfield_infra_deal_memo.txt` | Credit deal memo | March 2024 |

**Fictional entities**: NovaTech Financial Corp (NVTF), Meridian Capital Partners, Apex Holdings Inc (APX), Cornerstone Bank & Trust (CBT), Atlas Research Group, Greenfield Infrastructure Partners, Sentinel Risk Management Group.

## Evaluation QA Pairs (`eval/qa_pairs.json`)

38 question-answer pairs across three difficulty levels:

| Difficulty | Count | Description |
|------------|-------|-------------|
| `easy` | 18 | Single-fact lookup from one document |
| `medium` | 14 | Requires understanding context or combining sentences |
| `hard` | 6 | Multi-document synthesis or comparison |

**Categories covered**: financial facts, credit quality, guidance, M&A, fund performance, market data, macro outlook, multi-doc comparison, multi-doc synthesis.

### QA Pair Schema

```json
{
  "id": "string",
  "question": "string",
  "answer": "string",
  "source_doc": "string or [list of strings for multi-doc]",
  "relevant_chunks": ["list of exact excerpts from source docs"],
  "difficulty": "easy | medium | hard",
  "category": "string",
  "requires_multi_doc": true   // only present when true
}
```

The `relevant_chunks` field contains verbatim excerpts that a correct retrieval system should surface, making it usable for measuring retrieval precision and recall independently of generation quality.
