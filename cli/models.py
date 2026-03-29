from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel


class AnalystType(str, Enum):
    MARKET = "market"
    SOCIAL = "social"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
    POLICY = "policy"
    CAPITAL_FLOW = "capital_flow"
    SENTIMENT = "sentiment"
    SECTOR_THEME = "sector_theme"
