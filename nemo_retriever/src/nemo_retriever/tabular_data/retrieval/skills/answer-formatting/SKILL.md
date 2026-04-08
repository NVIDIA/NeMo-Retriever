---
name: answer-formatting
description: For formatting the final SQL-based answer into a consistent structured output with sql_code, answer, and result
---

Respond with **only** a single JSON object, no extra text.

Required shape:
- `sql_code`: exact SQL string that was executed (no markdown fences, no comments)
- `answer`: 1–3 sentences explaining what the result means for the user’s question
- `result`: raw database result (number, string, list of rows/objects, etc.; do not force to string)
