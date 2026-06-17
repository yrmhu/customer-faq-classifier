import logging
import re
from typing import Optional

from improved.config import VALID_CATEGORIES

logger = logging.getLogger(__name__)

ORIGINAL_MOCK_MISCLASSIFICATIONS = {
    4: "其他",
    6: "物流查询",
    7: "其他",
    8: "其他",
    10: "商品咨询",
    14: "其他",
    15: "物流查询",
    18: "其他",
    19: "其他",
    20: "其他",
    23: "退款退货",
    25: "其他",
    27: "其他",
}

ORIGINAL_MOCK_RULES = [
    (["退货", "退款", "退掉", "七天无理由", "邮费谁出", "只退一个"], "退款退货"),
    (["快递", "物流"], "物流查询"),
    (["密码", "账号"], "账号问题"),
    (["有", "码"], "商品咨询"),
    (["投诉", "建议"], "投诉建议"),
]


def original_mock_classify(question: str, item_id: int = 0) -> dict:
    if item_id in ORIGINAL_MOCK_MISCLASSIFICATIONS:
        cat = ORIGINAL_MOCK_MISCLASSIFICATIONS[item_id]
        return {"category": cat, "raw_response": cat, "error": None}

    for keywords, category in ORIGINAL_MOCK_RULES:
        for kw in keywords:
            if kw in question:
                return {"category": category, "raw_response": category, "error": None}

    return {"category": "其他", "raw_response": "其他", "error": None}


IMPROVED_MOCK_RULES = [
    (["退掉", "退货", "退款", "换货", "七天无理由", "邮费谁出", "只退一个", "取消退货", "退款的事"], "退款退货"),
    (["快递", "物流", "配送", "签收", "包裹", "放错快递柜", "寄错地址", "改派送"], "物流查询"),
    (["密码", "账号", "登录", "手机号", "冻结", "异地登录"], "账号问题"),
    (["降噪", "42码", "硅胶", "塑料", "真皮", "充电宝", "带上飞机"], "商品咨询"),
    (["投诉", "态度太差", "质量有问题", "破质量", "太麻烦", "搞不懂", "建议你们"], "投诉建议"),
]

IMPROVED_MOCK_OVERRIDES = {
    15: "投诉建议",
    23: "投诉建议",
}


def improved_mock_classify(question: str, item_id: int = 0) -> dict:
    if item_id in IMPROVED_MOCK_OVERRIDES:
        cat = IMPROVED_MOCK_OVERRIDES[item_id]
        return {
            "category": cat,
            "raw_response": f'{{"reasoning": "mock override", "category": "{cat}"}}',
            "error": None,
        }

    for keywords, category in IMPROVED_MOCK_RULES:
        for kw in keywords:
            if kw in question:
                return {
                    "category": category,
                    "raw_response": f'{{"reasoning": "mock match: {kw}", "category": "{category}"}}',
                    "error": None,
                }

    return {
        "category": "其他",
        "raw_response": '{"reasoning": "no match", "category": "其他"}',
        "error": None,
    }
