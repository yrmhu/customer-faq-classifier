import os

VALID_CATEGORIES = ["退款退货", "物流查询", "账号问题", "商品咨询", "投诉建议", "其他"]

MODEL = os.environ.get("CLASSIFIER_MODEL", "gpt-4o-mini")
MAX_RETRIES = int(os.environ.get("CLASSIFIER_MAX_RETRIES", "3"))
