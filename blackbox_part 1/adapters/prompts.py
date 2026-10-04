PLAN = """### PLAN
You are a research agent with two tools:
- search(query): looks up one fact, e.g. "population of France"
- calculator(expression): evaluates arithmetic, e.g. "12.5+3"

Task: {task}

Make a plan. Search for each fact you need (one search per fact, in the order the task
mentions them), then give ONE arithmetic expression that combines the facts, using
{{0}}, {{1}}, ... as placeholders for the numbers found by search 0, search 1, ...
Reply with JSON only, for example:
{{"steps": [{{"tool": "search", "query": "height of Mont Blanc"}}, {{"tool": "search", "query": "height of Ben Nevis"}}], "expression": "{{0}}-{{1}}"}}
"""

READ = """### READ
Extract the single number that answers the question from the text.
Reply with the number only (no units, no words). If there is no number, reply: unknown

Question: {question}
Text: {text}
"""

ANSWER = """### ANSWER
Task: {task}
Calculation result: {result}

Write the final answer as: Final answer: <number>
"""
