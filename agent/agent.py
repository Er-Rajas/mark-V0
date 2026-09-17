import json
import ollama

from agent.tool import TOOLS, FUNCTION_MAP


def parse_text_tool_calls(content):
    """
    Parse Phi-4 textual tool calls into a normalized format.

    Supported formats:

    1. [{"name": "get_overview", "arguments": {}}]

    2. [
           {
               "type": "function",
               "function": {
                   "name": "get_overview",
                   "arguments": {}
               }
           }
       ]

    3. functs[{"name": "get_overview", "arguments": {}}]

    4. functools[{"name": "get_overview", "arguments": {}}]
    """

    if not content:
        return []

    text = content.strip()

    # Remove markdown code fences
    text = text.replace("```json", "")
    text = text.replace("```", "")

    # Remove Phi-4 function markers
    text = text.replace("functs", "")
    text = text.replace("functools", "")

    text = text.strip()

    # Find JSON array
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1:
        return []

    json_text = text[start:end + 1]

    try:
        calls = json.loads(json_text)

    except json.JSONDecodeError:
        return []

    if not isinstance(calls, list):
        return []

    normalized = []

    for call in calls:

        if not isinstance(call, dict):
            continue

        # Format:
        # {"name": "...", "arguments": {...}}
        if "name" in call:

            normalized.append({
                "name": call["name"],
                "arguments": call.get("arguments", {})
            })

        # Format:
        # {"type": "function",
        #  "function": {
        #      "name": "...",
        #      "arguments": {}
        #  }}
        elif "function" in call:

            function = call["function"]

            if isinstance(function, dict) and "name" in function:

                normalized.append({
                    "name": function["name"],
                    "arguments": function.get("arguments", {})
                })

    return normalized


class MiniDataAnalyst:

    def __init__(self, df):
        self.df = df

    def run(self, query):

        messages = [
            {
                "role": "system",
                "content": """
You are Mini Data Analyst.

Perform only basic descriptive analysis.

You may ONLY use the provided tools.

The dataframe is held by the Python application.
Use the provided tools to obtain information about it.

Do not modify the dataframe.
Do not create plots.
Do not delete data.
Do not invent statistics.

When the user asks for dataset analysis,
use the available tools to obtain the required information.
"""
            },
            {
                "role": "user",
                "content": query
            }
        ]

        # Maximum number of agent iterations
        for step in range(5):

            response = ollama.chat(
                model="qwen3:4b ",
                messages=messages,
                tools=TOOLS
            )

            message = response.message

            print(f"\n===== STEP {step + 1} =====")

            print("\nMESSAGE:")
            print(message)

            print("\nNATIVE TOOL CALLS:")
            print(message.tool_calls)

            # ==================================================
            # CASE 1: Native Ollama tool call
            # ==================================================

            if message.tool_calls:

                messages.append(message)

                for tool_call in message.tool_calls:

                    tool_name = tool_call.function.name

                    print(f"\n🔧 Calling tool: {tool_name}")

                    # Safety check
                    if tool_name not in FUNCTION_MAP:

                        print(f"❌ Unknown tool: {tool_name}")

                        continue

                    function = FUNCTION_MAP[tool_name]

                    result = function(self.df)

                    print("\n📊 TOOL RESULT:")
                    print(result)

                    messages.append({
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": str(result)
                    })

                # Ask model what to do next
                continue

            # ==================================================
            # CASE 2: Phi-4 textual tool call
            # ==================================================

            text_tool_calls = parse_text_tool_calls(
                message.content
            )

            if text_tool_calls:

                print("\n⚠️ TEXTUAL TOOL CALL DETECTED:")
                print(text_tool_calls)

                # Preserve Phi-4's textual response
                messages.append({
                    "role": "assistant",
                    "content": message.content
                })

                for call in text_tool_calls:

                    tool_name = call["name"]

                    print(f"\n🔧 Calling tool: {tool_name}")

                    # Safety check
                    if tool_name not in FUNCTION_MAP:

                        print(f"❌ Unknown tool: {tool_name}")

                        continue

                    function = FUNCTION_MAP[tool_name]

                    result = function(self.df)

                    print("\n📊 TOOL RESULT:")
                    print(result)

                    messages.append({
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": str(result)
                    })

                # Continue the agent loop
                continue

            # ==================================================
            # CASE 3: No tool call
            # ==================================================

            print("\n✅ FINAL RESPONSE GENERATED")

            return message.content

        # ======================================================
        # Maximum steps reached
        # ======================================================

        return "Agent stopped: maximum steps reached."  