import ollama

from agent.tool import TOOLS, FUNCTION_MAP


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

                Do not modify the dataframe.
                Do not create plots.
                Do not delete data.
                Do not invent statistics.
                """
            },
            {
                "role": "user",
                "content": query
            }
        ]

        for _ in range(5):

            response = ollama.chat(
                model="qwen3:4b",
                messages=messages,
                tools=TOOLS
            )

            messages.append(response.message)

            # No tools requested = final answer
            if not response.message.tool_calls:
                return response.message.content

            # Execute tools
            for tool_call in response.message.tool_calls:

                tool_name = tool_call.function.name

                print(f"🔧 Calling tool: {tool_name}")

                function = FUNCTION_MAP[tool_name]

                result = function(self.df)

                messages.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": str(result)
                })

        return "Agent stopped: maximum steps reached."