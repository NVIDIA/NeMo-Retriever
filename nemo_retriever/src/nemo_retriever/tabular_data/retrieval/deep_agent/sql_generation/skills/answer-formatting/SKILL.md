---
name: answer-formatting
description: How to format the mandatory final JSON answer after SQL execution.
---

# Answer Formatting Skill

## Purpose

Your final assistant turn **must** be a single JSON object and nothing else.
This skill defines exactly how to construct it.

---

## Required JSON Schema

```json
{
  "sql_code":         "<exact SQL string you passed to execute_sql>",
  "answer":           "<1–3 sentence plain-text answer to the user question>",
  "result":           <number | string | list-of-row-dicts | null>,
  "semantic_elements": [
    {"id": "<candidate-id>", "label": "<label>", "classification": true}
  ]
}
```

### Field rules

| Field | Type | Rule |
|-------|------|------|
| `sql_code` | string | The exact SQL passed to `execute_sql`. No markdown fences inside the value. If no SQL was generated, use `""`. |
| `answer` | string | Plain text, 1–3 sentences. Directly answers the user question using the DB result. |
| `result` | any | The `result` value from `execute_sql` response. Use `null` if execution was skipped or returned nothing. |
| `semantic_elements` | array | Each custom analysis / certified candidate referenced while building the SQL. Use `[]` if none were used. |

---

## Formatting Rules

1. **Nothing before `{`** — no greeting, no preamble, no "Here is the answer:".
2. **Nothing after `}`** — no postscript, no apology, no "I hope this helps".
3. **No markdown fences** — do NOT wrap the JSON in ` ```json ``` `.
4. **Valid JSON only** — ensure property names are double-quoted and the object is parseable.
5. **Do NOT include `sql_code` value inside markdown fences** — the value is a plain JSON string; any backticks inside would be literal characters.

---

## Examples

### Successful execution

```
{"sql_code":"SELECT COUNT(*) AS total_orders FROM SALES.ORDERS WHERE YEAR(OrderDate)=2023","answer":"There were 4,821 orders placed in 2023.","result":4821,"semantic_elements":[]}
```

### Execution returned multiple rows

```
{"sql_code":"SELECT ProductName, SUM(Revenue) AS total FROM SALES.INVOICELINES GROUP BY ProductName ORDER BY total DESC LIMIT 5","answer":"The top 5 products by revenue in 2023 were Widget A ($120k), Widget B ($95k), Widget C ($87k), Widget D ($76k), and Widget E ($64k).","result":[{"ProductName":"Widget A","total":120000},{"ProductName":"Widget B","total":95000}],"semantic_elements":[{"id":"abc123","label":"custom_analysis","classification":true}]}
```

### SQL could not be constructed

```
{"sql_code":"","answer":"I could not find tables relevant to your question about supplier delivery times. Please check that the relevant data has been ingested, or rephrase the question.","result":null,"semantic_elements":[]}
```

---

## When to Emit the Final Answer

Emit the JSON object once ALL of the following are true:
- You have called `execute_sql` (or determined SQL is unconstructable).
- You have a final `sql_code` value (may be `""`).
- You have composed a clear `answer` sentence.
- You have collected the `result` from the execution response.

Do NOT emit the JSON prematurely — finish the full retrieval → generation → validation →
execution loop first.
