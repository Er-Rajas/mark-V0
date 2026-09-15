from tools.statistics import numerical_summary
from tools.overview import get_overview


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_overview",
            "description": "Get basic information about the dataset.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "numerical_summary",
            "description": "Get descriptive statistics for numerical columns.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

FUNCTION_MAP = {
    "get_overview": get_overview,
    "numerical_summary": numerical_summary,
}