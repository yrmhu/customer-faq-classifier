#!/usr/bin/env python3
"""
评估脚本：对比改进前后分类准确率
支持 mock 模式和真实 API 模式
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from improved.classifier import classify_question
from improved.client import get_client
from improved.config import VALID_CATEGORIES
from improved.mock import original_mock_classify, improved_mock_classify
from improved.prompts import ORIGINAL_PROMPT_TEMPLATE, SYSTEM_PROMPT_V2
from improved.parser import parse_category


def run_evaluation(test_file: str, classify_fn, version_name: str) -> dict:
    with open(test_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    results = []
    correct = 0
    total = len(samples)
    errors_by_category = {}
    category_stats = {}

    for item in samples:
        question = item["question"]
        true_label = item["label"]

        result = classify_fn(question, item_id=item["id"])
        predicted = result["category"]

        is_correct = predicted == true_label
        if is_correct:
            correct += 1
        else:
            if true_label not in errors_by_category:
                errors_by_category[true_label] = []
            errors_by_category[true_label].append({
                "id": item["id"],
                "question": question,
                "predicted": predicted,
                "true": true_label,
            })

        if true_label not in category_stats:
            category_stats[true_label] = {"total": 0, "correct": 0}
        category_stats[true_label]["total"] += 1
        if is_correct:
            category_stats[true_label]["correct"] += 1

        results.append({
            "id": item["id"],
            "question": question,
            "true_category": true_label,
            "predicted_category": predicted,
            "correct": is_correct,
        })

    accuracy = correct / total * 100 if total > 0 else 0

    per_category = {}
    for cat, stats in category_stats.items():
        per_category[cat] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0,
        }

    return {
        "version": version_name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_category": per_category,
        "errors_by_category": errors_by_category,
        "details": results,
    }


def print_comparison(original_result: dict, improved_result: dict):
    print("\n" + "=" * 70)
    print("                    EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Original':<20} {'Improved':<20} {'Delta'}")
    print("-" * 70)
    print(f"{'Total':<20} {original_result['total']:<20} {improved_result['total']:<20}")
    print(f"{'Correct':<20} {original_result['correct']:<20} {improved_result['correct']:<20} {'+' + str(improved_result['correct'] - original_result['correct'])}")
    delta = improved_result['accuracy'] - original_result['accuracy']
    sign = '+' if delta > 0 else ''
    print(f"{'Accuracy':<20} {original_result['accuracy']:.1f}%{'':<15} {improved_result['accuracy']:.1f}%{'':<15} {sign}{delta:.1f}%")

    print("\n" + "-" * 70)
    print("Per-category accuracy:")
    print("-" * 70)
    all_cats = sorted(set(list(original_result["per_category"].keys()) + list(improved_result["per_category"].keys())))
    print(f"{'Category':<12} {'Orig Acc':<12} {'Impr Acc':<12} {'Delta'}")
    for cat in all_cats:
        orig_acc = original_result["per_category"].get(cat, {}).get("accuracy", 0)
        impr_acc = improved_result["per_category"].get(cat, {}).get("accuracy", 0)
        d = impr_acc - orig_acc
        s = "+" if d > 0 else ""
        print(f"{cat:<12} {orig_acc:.0f}%{'':<8} {impr_acc:.0f}%{'':<8} {s}{d:.0f}%")

    print("\n" + "-" * 70)
    print("Errors in improved version:")
    print("-" * 70)
    if improved_result["errors_by_category"]:
        for cat, errors in improved_result["errors_by_category"].items():
            print(f"\n  [{cat}] {len(errors)} error(s):")
            for e in errors:
                print(f"    #{e['id']}: \"{e['question'][:35]}\" -> predicted: {e['predicted']} (true: {e['true']})")
    else:
        print("  All correct!")

    print("\n" + "-" * 70)
    print("Detailed comparison:")
    print("-" * 70)
    print(f"{'ID':<4} {'Question':<35} {'True':<8} {'Orig':<8} {'Impr':<8} {'O':<3} {'I':<3}")
    for orig, impr in zip(original_result["details"], improved_result["details"]):
        q = orig["question"][:33]
        o_mark = "Y" if orig["correct"] else "N"
        i_mark = "Y" if impr["correct"] else "N"
        print(f"{orig['id']:<4} {q:<35} {orig['true_category']:<8} {orig['predicted_category']:<8} {impr['predicted_category']:<8} {o_mark:<3} {i_mark:<3}")

    print("\n" + "=" * 70)


def main():
    test_file = os.path.join(os.path.dirname(__file__), "test_samples.json")
    use_mock = "--mock" in sys.argv

    if use_mock:
        print("Using MOCK mode (no API Key required)")
        original_fn = original_mock_classify
        improved_fn = improved_mock_classify
    else:
        print("Using REAL API mode")
        try:
            client = get_client()

            def original_fn(question, item_id=0):
                prompt = ORIGINAL_PROMPT_TEMPLATE.format(question=question)
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                    )
                    raw = response.choices[0].message.content.strip()
                    category = parse_category(raw)
                    if category is None:
                        category = "其他"
                    return {"category": category, "raw_response": raw, "error": None}
                except Exception as e:
                    return {"category": "其他", "raw_response": "", "error": str(e)}

            def improved_fn(question, item_id=0):
                return classify_question(question, client=client, use_mock=False, item_id=item_id)

        except Exception as e:
            print(f"API init failed: {e}")
            print("Falling back to MOCK mode")
            original_fn = original_mock_classify
            improved_fn = improved_mock_classify

    print("\nEvaluating original version...")
    original_result = run_evaluation(test_file, original_fn, "Original")

    print("Evaluating improved version...")
    improved_result = run_evaluation(test_file, improved_fn, "Improved")

    print_comparison(original_result, improved_result)

    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "original_result.json"), "w", encoding="utf-8") as f:
        json.dump(original_result, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "improved_result.json"), "w", encoding="utf-8") as f:
        json.dump(improved_result, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
