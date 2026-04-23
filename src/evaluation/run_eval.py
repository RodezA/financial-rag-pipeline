import json
from pathlib import Path

import anthropic

from src.evaluation.metrics import (
    answer_quality_score,
    faithfulness_score,
    precision_at_k,
    recall_at_k,
)
from src.pipeline import RAGPipeline
from src.utils import load_config


def run_evaluation():
    cfg = load_config()
    qa_path = Path(cfg["paths"]["eval_data"])
    output_path = Path(cfg["paths"]["eval_output"])

    with open(qa_path) as f:
        qa_pairs = json.load(f)

    pipeline = RAGPipeline(use_reranker=True)
    judge_client = anthropic.Anthropic()
    judge_model = cfg["generation"]["model"]

    results = []
    for i, qa in enumerate(qa_pairs):
        print(f"[{i + 1}/{len(qa_pairs)}] {qa['id']}: {qa['question'][:60]}...")
        try:
            pipeline_result = pipeline.query(qa["question"])
            retrieved = pipeline_result["source_docs"]

            relevant = qa["relevant_chunks"]
            if isinstance(relevant, str):
                relevant = [relevant]

            prec = precision_at_k(retrieved, relevant)
            rec = recall_at_k(retrieved, relevant)
            faith = faithfulness_score(
                pipeline_result["answer"], retrieved, judge_client, judge_model
            )
            quality = answer_quality_score(
                pipeline_result["answer"], qa["answer"], judge_client, judge_model
            )

            results.append(
                {
                    "id": qa["id"],
                    "difficulty": qa.get("difficulty", "unknown"),
                    "category": qa.get("category", "unknown"),
                    "question": qa["question"],
                    "ground_truth": qa["answer"],
                    "generated_answer": pipeline_result["answer"],
                    "precision_at_k": prec,
                    "recall_at_k": rec,
                    "faithfulness": faith,
                    "answer_quality": quality,
                    "latency_ms": pipeline_result["latency_ms"],
                    "token_usage": pipeline_result["token_usage"],
                }
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"id": qa["id"], "error": str(e)})

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No valid results to summarize.")
        return

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Questions evaluated: {len(valid)}/{len(qa_pairs)}")
    print(f"Avg Precision@k:    {sum(r['precision_at_k'] for r in valid) / len(valid):.3f}")
    print(f"Avg Recall@k:       {sum(r['recall_at_k'] for r in valid) / len(valid):.3f}")
    print(f"Avg Faithfulness:   {sum(r['faithfulness'] for r in valid) / len(valid):.2f}/5")
    print(f"Avg Answer Quality: {sum(r['answer_quality'] for r in valid) / len(valid):.2f}/5")
    print(f"Avg Latency:        {sum(r['latency_ms'] for r in valid) / len(valid):.0f}ms")
    print(f"\nResults saved to: {output_path}")

    # Per-difficulty breakdown
    for level in ("easy", "medium", "hard"):
        subset = [r for r in valid if r.get("difficulty") == level]
        if subset:
            print(
                f"\n  {level.capitalize()} ({len(subset)} questions): "
                f"P={sum(r['precision_at_k'] for r in subset)/len(subset):.3f}  "
                f"R={sum(r['recall_at_k'] for r in subset)/len(subset):.3f}  "
                f"F={sum(r['faithfulness'] for r in subset)/len(subset):.2f}  "
                f"Q={sum(r['answer_quality'] for r in subset)/len(subset):.2f}"
            )


if __name__ == "__main__":
    run_evaluation()
