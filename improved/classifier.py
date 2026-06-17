#!/usr/bin/env python3
"""
客服 FAQ 自动分类脚本（改进版 v2）
改进点：
1. API Key 从环境变量读取，不再硬编码
2. 使用 OpenAI v1.x 客户端风格
3. 完善错误处理 + 指数退避重试
4. 输出验证：校验返回的类别是否合法
5. 日志记录
6. 改进 Prompt（system prompt + 类别定义 + Few-shot + JSON 输出 + 思维链）
7. 异步并发支持
8. 模块化代码结构
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional

from improved.client import get_client
from improved.config import MAX_RETRIES, MODEL, VALID_CATEGORIES
from improved.mock import improved_mock_classify, original_mock_classify
from improved.parser import parse_category
from improved.prompts import SYSTEM_PROMPT_V2, build_user_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def classify_question(
    question: str,
    client=None,
    model: str = MODEL,
    max_retries: int = MAX_RETRIES,
    use_mock: bool = False,
    item_id: int = 0,
) -> dict:
    if use_mock:
        return improved_mock_classify(question, item_id=item_id)

    if client is None:
        client = get_client()

    user_message = build_user_message(question)

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_V2},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
            )

            if not response.choices:
                logger.warning("API returned empty choices")
                last_error = "Empty choices from API"
                wait = 2 ** attempt
                time.sleep(wait)
                continue

            content = response.choices[0].message.content
            if content is None:
                logger.warning("API returned None content (possibly content filter)")
                last_error = "None content from API"
                wait = 2 ** attempt
                time.sleep(wait)
                continue

            raw = content.strip()
            if not raw:
                logger.warning("API returned empty content")
                last_error = "Empty content from API"
                wait = 2 ** attempt
                time.sleep(wait)
                continue

            category = parse_category(raw)

            if category is None:
                logger.warning(f"Cannot parse category from: {raw}")
                category = "其他"

            return {"category": category, "raw_response": raw, "error": None}

        except Exception as e:
            last_error = str(e)
            wait = 2 ** attempt
            logger.warning(f"Attempt {attempt + 1} failed: {e}, retry in {wait}s...")
            time.sleep(wait)

    logger.error(f"Max retries reached: {last_error}")
    return {"category": "其他", "raw_response": "", "error": last_error}


async def classify_question_async(
    question: str,
    client=None,
    model: str = MODEL,
    max_retries: int = MAX_RETRIES,
    use_mock: bool = False,
    item_id: int = 0,
) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: classify_question(question, client, model, max_retries, use_mock, item_id),
    )


async def batch_classify_async(
    questions: list,
    model: str = MODEL,
    use_mock: bool = False,
    concurrency: int = 5,
) -> list:
    client = None
    if not use_mock:
        client = get_client()

    semaphore = asyncio.Semaphore(concurrency)

    async def _classify_one(item):
        async with semaphore:
            return await classify_question_async(
                item["question"],
                client=client,
                model=model,
                use_mock=use_mock,
                item_id=item.get("id", 0),
            )

    tasks = [_classify_one(item) for item in questions]
    results_raw = await asyncio.gather(*tasks)

    results = []
    for item, result in zip(questions, results_raw):
        predicted = result["category"]
        entry = {
            "id": item["id"],
            "question": item["question"],
            "predicted_category": predicted,
        }
        label = item.get("label")
        if label:
            entry["true_category"] = label
            entry["correct"] = predicted == label
        if result["error"]:
            entry["error"] = result["error"]
        results.append(entry)

    return results


def batch_classify(
    input_file: str,
    output_file: str,
    model: str = MODEL,
    use_mock: bool = False,
    async_mode: bool = False,
    concurrency: int = 5,
):
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            questions = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {e}")
        raise ValueError(f"Invalid JSON in input file: {e}")

    if not isinstance(questions, list):
        logger.error("Input JSON must be a list of objects")
        raise ValueError("Input JSON must be a list of objects")

    results = []

    def _save_partial():
        if results:
            partial_file = output_file + ".partial"
            with open(partial_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Partial results saved to {partial_file} ({len(results)} items)")

    if async_mode and not use_mock:
        try:
            results = asyncio.run(batch_classify_async(questions, model, use_mock, concurrency))
        except Exception as e:
            logger.error(f"Async batch classify failed: {e}")
            _save_partial()
            raise
    else:
        client = None
        if not use_mock:
            client = get_client()

        for item in questions:
            if not isinstance(item, dict) or "question" not in item:
                logger.warning(f"Skipping invalid item: {item}")
                continue

            try:
                result = classify_question(
                    item["question"],
                    client=client,
                    model=model,
                    use_mock=use_mock,
                    item_id=item.get("id", 0),
                )
                predicted = result["category"]
                entry = {
                    "id": item.get("id", len(results) + 1),
                    "question": item["question"],
                    "predicted_category": predicted,
                }
                label = item.get("label")
                if label:
                    entry["true_category"] = label
                    entry["correct"] = predicted == label
                if result["error"]:
                    entry["error"] = result["error"]
                results.append(entry)
                logger.info(
                    f"[{entry['id']}] {item['question'][:30]}... -> {predicted}"
                    + (f" (true: {label})" if label else "")
                )
            except Exception as e:
                logger.error(f"Failed to classify item {item}: {e}")
                results.append({
                    "id": item.get("id", len(results) + 1),
                    "question": item.get("question", ""),
                    "predicted_category": "其他",
                    "error": str(e),
                })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    labeled = [r for r in results if "true_category" in r]
    if labeled:
        correct = sum(1 for r in labeled if r["correct"])
        total = len(labeled)
        accuracy = correct / total * 100
        logger.info(f"Done: {len(results)} items, accuracy: {accuracy:.1f}% ({correct}/{total})")
    else:
        logger.info(f"Done: {len(results)} items")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python classifier.py <input> <output> [--mock] [--async] [--model MODEL]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    use_mock = "--mock" in sys.argv
    async_mode = "--async" in sys.argv
    model = MODEL

    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model = sys.argv[idx + 1]

    batch_classify(input_file, output_file, model=model, use_mock=use_mock, async_mode=async_mode)
