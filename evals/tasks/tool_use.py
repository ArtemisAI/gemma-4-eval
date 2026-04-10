"""Tool calling eval tasks.

Tests function calling, JSON parameter handling, and delimiter correctness.
Specifically targets known llama.cpp issues:
- #21375: Infinite repetition loop with peg-gemma4 parser
- #21384: Array params serialized as JSON string with { or }
- #21316: Unexpected tokens after tool calls
"""


def get_tasks() -> list[dict]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search a database with filters",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string"},
                                    "operator": {"type": "string", "enum": ["eq", "gt", "lt", "contains"]},
                                    "value": {"type": "string"},
                                },
                            },
                            "description": "Array of filter objects",
                        },
                        "limit": {"type": "integer", "description": "Max results"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_event",
                "description": "Create a calendar event",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "start_time": {"type": "string", "description": "ISO 8601 datetime"},
                        "end_time": {"type": "string", "description": "ISO 8601 datetime"},
                        "attendees": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of attendee emails",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Arbitrary key-value metadata",
                        },
                    },
                    "required": ["title", "start_time", "end_time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_query",
                "description": "Execute a SQL query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "SQL query to execute"},
                        "params": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Parameterized query values",
                        },
                    },
                    "required": ["sql"],
                },
            },
        },
    ]

    return [
        {
            "name": "simple_tool_call",
            "messages": [
                {
                    "role": "user",
                    "content": "Search the database for users named 'Alice' with age greater than 30, limit to 5 results.",
                }
            ],
            "tools": tools,
        },
        {
            "name": "array_params_with_braces",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Search the database for records where the 'config' field contains "
                        "'{\"type\": \"premium\"}' and the 'tags' field contains 'v2.{latest}'. "
                        "Also filter where status equals 'active'. Limit to 10."
                    ),
                }
            ],
            "tools": tools,
            "_note": "Tests #21384: array params with { } characters",
        },
        {
            "name": "multi_tool_sequential",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I need you to: 1) Search the database for all meetings scheduled today, "
                        "2) Create a new event titled 'Team Sync' from 3pm to 4pm today with "
                        "attendees alice@co.com, bob@co.com, and metadata {\"priority\": \"high\", "
                        "\"project\": \"gemma-eval\"}, 3) Execute a SQL query to update the "
                        "meetings table: UPDATE meetings SET status='confirmed' WHERE id IN (SELECT id "
                        "FROM meetings WHERE date=CURRENT_DATE AND status='pending')"
                    ),
                }
            ],
            "tools": tools,
            "_note": "Tests multi-tool in one turn + nested JSON in metadata",
        },
        {
            "name": "tool_with_reasoning",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I have a database of customer orders. I need to find all orders from Q1 2024 "
                        "where the total exceeds $1000 and the customer is from California. Think about "
                        "the most efficient query approach, then execute it. Also search the database "
                        "for any related returns or refunds for those orders."
                    ),
                }
            ],
            "tools": tools,
            "_note": "Tests thinking + tool calling interaction (triggers #21375 on some backends)",
        },
    ]
