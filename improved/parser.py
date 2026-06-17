import json
import re
from typing import Optional

from improved.config import VALID_CATEGORIES


def parse_category(response_text: str) -> Optional[str]:
    text = response_text.strip()

    json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            category = data.get("category", "").strip()
            if category in VALID_CATEGORIES:
                return category
        except json.JSONDecodeError:
            pass

    for cat in VALID_CATEGORIES:
        if cat in text:
            return cat

    return None
