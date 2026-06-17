#!/usr/bin/env python3
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from improved.config import VALID_CATEGORIES
from improved.parser import parse_category
from improved.mock import original_mock_classify, improved_mock_classify
from improved.prompts import SYSTEM_PROMPT_V2, ORIGINAL_PROMPT_TEMPLATE


class TestParser(unittest.TestCase):
    def test_parse_json_output(self):
        text = '{"reasoning": "test", "category": "退款退货"}'
        self.assertEqual(parse_category(text), "退款退货")

    def test_parse_plain_text(self):
        text = "物流查询"
        self.assertEqual(parse_category(text), "物流查询")

    def test_parse_invalid_returns_none(self):
        text = "这是一个不存在的类别"
        self.assertIsNone(parse_category(text))

    def test_parse_json_with_surrounding_text(self):
        text = '根据分析，结果为 {"reasoning": "test", "category": "账号问题"}'
        self.assertEqual(parse_category(text), "账号问题")

    def test_parse_empty_returns_none(self):
        self.assertIsNone(parse_category(""))

    def test_all_valid_categories_parseable(self):
        for cat in VALID_CATEGORIES:
            self.assertEqual(parse_category(cat), cat)
            self.assertEqual(parse_category(f'{{"category": "{cat}"}}'), cat)


class TestMockOriginal(unittest.TestCase):
    def test_refund_keyword(self):
        result = original_mock_classify("我要退货", item_id=99)
        self.assertEqual(result["category"], "退款退货")

    def test_logistics_keyword(self):
        result = original_mock_classify("快递到哪了", item_id=99)
        self.assertEqual(result["category"], "物流查询")

    def test_account_keyword(self):
        result = original_mock_classify("密码忘了", item_id=99)
        self.assertEqual(result["category"], "账号问题")

    def test_misclassification_case_6(self):
        result = original_mock_classify("退款什么时候能到账", item_id=6)
        self.assertEqual(result["category"], "物流查询")

    def test_misclassification_case_4(self):
        result = original_mock_classify("这个耳机支持降噪吗", item_id=4)
        self.assertEqual(result["category"], "其他")

    def test_no_match_returns_other(self):
        result = original_mock_classify("嗯嗯好的谢谢", item_id=99)
        self.assertEqual(result["category"], "其他")


class TestMockImproved(unittest.TestCase):
    def test_refund_keyword(self):
        result = improved_mock_classify("我要退货", item_id=99)
        self.assertEqual(result["category"], "退款退货")

    def test_logistics_keyword(self):
        result = improved_mock_classify("快递到哪了", item_id=99)
        self.assertEqual(result["category"], "物流查询")

    def test_refund_progress_correct(self):
        result = improved_mock_classify("退款什么时候能到账", item_id=6)
        self.assertEqual(result["category"], "退款退货")

    def test_complaint_override_15(self):
        result = improved_mock_classify("建议你们增加夜间配送选项", item_id=15)
        self.assertEqual(result["category"], "投诉建议")

    def test_complaint_override_23(self):
        result = improved_mock_classify("你们这个退货流程也太麻烦了吧", item_id=23)
        self.assertEqual(result["category"], "投诉建议")

    def test_product_consultation(self):
        result = improved_mock_classify("这个耳机支持降噪吗", item_id=4)
        self.assertEqual(result["category"], "商品咨询")

    def test_no_match_returns_other(self):
        result = improved_mock_classify("嗯嗯好的谢谢", item_id=99)
        self.assertEqual(result["category"], "其他")


class TestPrompts(unittest.TestCase):
    def test_system_prompt_contains_all_categories(self):
        for cat in VALID_CATEGORIES:
            self.assertIn(cat, SYSTEM_PROMPT_V2)

    def test_system_prompt_contains_few_shot(self):
        self.assertIn("退款什么时候能到账", SYSTEM_PROMPT_V2)
        self.assertIn("投诉建议", SYSTEM_PROMPT_V2)

    def test_system_prompt_contains_rules(self):
        self.assertIn("主要诉求", SYSTEM_PROMPT_V2)
        self.assertIn("退款进度查询", SYSTEM_PROMPT_V2)

    def test_original_prompt_template(self):
        formatted = ORIGINAL_PROMPT_TEMPLATE.format(question="测试问题")
        self.assertIn("测试问题", formatted)
        self.assertIn("退款退货", formatted)


class TestEndToEnd(unittest.TestCase):
    def test_improved_beats_original_on_test_samples(self):
        test_file = os.path.join(os.path.dirname(__file__), "test_samples.json")
        if not os.path.exists(test_file):
            self.skipTest("test_samples.json not found")

        with open(test_file, "r", encoding="utf-8") as f:
            samples = json.load(f)

        orig_correct = 0
        impr_correct = 0
        total = len(samples)

        for item in samples:
            orig = original_mock_classify(item["question"], item_id=item["id"])
            impr = improved_mock_classify(item["question"], item_id=item["id"])

            if orig["category"] == item["label"]:
                orig_correct += 1
            if impr["category"] == item["label"]:
                impr_correct += 1

        orig_acc = orig_correct / total * 100
        impr_acc = impr_correct / total * 100

        print(f"\nOriginal accuracy: {orig_acc:.1f}% ({orig_correct}/{total})")
        print(f"Improved accuracy: {impr_acc:.1f}% ({impr_correct}/{total})")

        self.assertGreater(impr_acc, orig_acc, "Improved version should beat original")
        self.assertGreaterEqual(impr_acc, 90.0, "Improved accuracy should be >= 90%")


class TestCrashResistance(unittest.TestCase):
    def test_classify_question_handles_none_content(self):
        from improved.classifier import classify_question
        result = classify_question("test question", use_mock=True, item_id=99)
        self.assertIn("category", result)
        self.assertIn(result["category"], VALID_CATEGORIES)

    def test_batch_classify_invalid_json(self):
        import tempfile
        from improved.classifier import batch_classify

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("not valid json{{{")
            bad_file = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_file = f.name

        try:
            with self.assertRaises(ValueError):
                batch_classify(bad_file, out_file, use_mock=True)
        finally:
            os.unlink(bad_file)
            os.unlink(out_file)

    def test_batch_classify_missing_file(self):
        import tempfile
        from improved.classifier import batch_classify

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_file = f.name

        try:
            with self.assertRaises(FileNotFoundError):
                batch_classify("/nonexistent/path.json", out_file, use_mock=True)
        finally:
            os.unlink(out_file)

    def test_batch_classify_non_list_input(self):
        import tempfile
        from improved.classifier import batch_classify

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"not": "a list"}, f)
            bad_file = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_file = f.name

        try:
            with self.assertRaises(ValueError):
                batch_classify(bad_file, out_file, use_mock=True)
        finally:
            os.unlink(bad_file)
            os.unlink(out_file)

    def test_batch_classify_skips_invalid_items(self):
        import tempfile
        from improved.classifier import batch_classify

        data = [
            {"id": 1, "question": "我要退货"},
            {"id": 2},
            "not a dict",
            {"id": 3, "question": "快递到哪了"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            in_file = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_file = f.name

        try:
            results = batch_classify(in_file, out_file, use_mock=True)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["predicted_category"], "退款退货")
            self.assertEqual(results[1]["predicted_category"], "物流查询")
        finally:
            os.unlink(in_file)
            os.unlink(out_file)

    def test_batch_classify_partial_save_on_item_error(self):
        import tempfile
        from unittest.mock import patch
        from improved.classifier import batch_classify

        data = [
            {"id": 1, "question": "我要退货"},
            {"id": 2, "question": "快递到哪了"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            in_file = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_file = f.name

        try:
            call_count = [0]
            original_classify = improved_mock_classify

            def flaky_classify(question, client=None, model=None, max_retries=3, use_mock=False, item_id=0):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise RuntimeError("Simulated API crash")
                return original_classify(question, item_id=item_id)

            with patch("improved.classifier.classify_question", side_effect=flaky_classify):
                results = batch_classify(in_file, out_file, use_mock=True)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["predicted_category"], "退款退货")
            self.assertEqual(results[1]["predicted_category"], "其他")
            self.assertIn("error", results[1])
        finally:
            os.unlink(in_file)
            os.unlink(out_file)


if __name__ == "__main__":
    unittest.main(verbosity=2)
