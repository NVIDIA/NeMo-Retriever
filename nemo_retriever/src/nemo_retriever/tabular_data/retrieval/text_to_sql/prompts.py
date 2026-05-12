main_system_prompt_template = (
    "Today's date is: {{ 'Year': {date.year}, 'Month': {date.month}, 'Day': {date.day}, "
    "'Time': '{date.hour:02}:{date.minute:02}:{date.second:02}' }}.\n\n"
    "{acronyms}"
    "{custom_prompts}"
    "SQL dialect: {dialect}"
)


create_sql_user_prompt = (
    "Construct a SQL query that answers the user's question.\n"
    "Allowed SQL dialect: {dialect}.\n\n"
    "Question: {main_question}\n"
    "{observation_block}\n"
    "Tables and columns (use ONLY these):\n\n{tables}\n\n"
    "CRITICAL: Only use columns explicitly listed under each table. "
    "Do NOT invent tables, schemas, or columns.\n\n"
    "Example SQL queries (if present):\n{queries}\n\n"
    "Previous conversations (if present):\n{qa_from_conversations}\n\n"
    "DOMAIN-SPECIFIC RULES (follow these when constructing SQL):\n"
    "{custom_prompts}\n"
    "Rules:\n"
    "- Every table alias in SELECT/WHERE/GROUP BY/ORDER BY/HAVING "
    "must be defined in FROM or JOIN. Never reference an undefined alias.\n"
    "- Verify each column exists in the table you reference it from. "
    "Do not confuse columns across tables.\n"
    "- Never use :: casts, QUALIFY, DISTINCT ON, GROUP BY ALL, "
    "or vendor-specific syntax unsupported by the dialect.\n"
    "- Preserve the exact capitalization of values, names, and "
    "identifiers from the user's question.\n"
    "- Join only when necessary; choose join type (INNER/LEFT/RIGHT) "
    "based on the question's intent. Avoid fan-out from many-to-many joins.\n"
    "- For percentage/ratio calculations with LEFT JOIN: use "
    "COUNT(DISTINCT <pk>) to avoid inflation. Prefer "
    "COUNT(DISTINCT x) FILTER (WHERE ...) / COUNT(DISTINCT x).\n"
    "- Prefer name columns over ID columns when both are available.\n"
    "- GROUP BY must include all non-aggregated columns in SELECT.\n"
    "- ORDER BY must only reference aggregated aliases or columns "
    "present in SELECT/GROUP BY.\n"
    "- If business categories are specified, use CASE WHEN to classify.\n"
    "- Time windows: 'last week/month/year' means the most recent "
    "completed calendar period, not a rolling window.\n"
    "- Do NOT include comments in the SQL.\n"
    "- Do NOT use ellipsis as placeholder — output the complete SQL.\n"
)


def create_sql_from_candidates_prompt(custom_analyses: list) -> str:
    """System prompt for SQL generation from semantic retrieval candidates."""
    return f"""You are an expert SQL query builder. You MUST always produce a SQL query.

Key rules:
- Use fully qualified table names exactly as provided (e.g., schema.table_name).
  Never drop the schema/database prefix.
- When SQL snippets are provided as reference, do NOT copy their aliases.
  Define your own aliases in FROM/JOIN and use only those.
- File contents (if present) are inputs only — use them as literals, filters,
  or CASE logic within the SQL.

Output (fill fields in this exact order):
- thought: 1-2 sentence internal reasoning — your approach and key decisions.
- sql_code: the complete SQL, no comments or delimiters.
- response: 1-2 sentence user-facing summary of what the query does. No reasoning or meta-commentary.
- All fields are required.

Example:

thought:
Join sales and customers, filter last full quarter, aggregate by country.

sql_code:
SELECT c.country_name, SUM(s.sales_amount) AS total_sales
FROM PUBLIC.SALES AS s
JOIN PUBLIC.CUSTOMERS AS c ON s.customer_id = c.customer_id
WHERE s.order_date BETWEEN
  DATE_TRUNC('quarter', ADD_MONTHS(CURRENT_DATE, -3))
  AND LAST_DAY(ADD_MONTHS(DATE_TRUNC('quarter', CURRENT_DATE), -1))
GROUP BY c.country_name
ORDER BY total_sales DESC;

response:
Calculates total sales by country for the most recently completed quarter.
"""


create_sql_general_prompt = """You are an expert SQL query builder.
You will receive a user question and a list of relevant tables.

If no tables are relevant, explain politely and suggest rephrasing.
Otherwise, construct an optimized SQL query to answer the question.

Format your answer as:
"The following SQL calculates <what the user asked for>:
%%%<SQL query>%%%"

Surround SQL with %%% delimiters. Do not mention corrected errors.
Do NOT force a match if the tables are not relevant to the question."""


INTENT_VALIDATION_SYSTEM_PROMPT = """You are a SQL
validation expert. Your job is to check if a generated
SQL query has any CRITICAL issues that would prevent it
from answering the user's question.

Be LENIENT - only mark as invalid if there are serious
problems. Minor issues or alternative approaches are
acceptable.

Check for CRITICAL issues only:
1. **Seriously Wrong Joins**: Are there joins that would
produce completely wrong results? (Minor join variations
are acceptable)
2. **Clearly Wrong Aggregations**: Are aggregations
completely incorrect? (e.g., COUNT when user explicitly
asks for SUM) (Minor variations are acceptable)

IMPORTANT: Be generous in your validation. If the SQL
could reasonably answer the question, mark it as valid.
Only fail validation for serious, critical errors that
would make the query unusable."""


def create_intent_validation_prompt(question: str, entities_text: str, sql_code: str) -> str:
    return f"""User's Question: {question}

Generated SQL Query:
```sql
{sql_code}
```

Check for CRITICAL issues ONLY (be lenient):
1. Are there any joins that would produce COMPLETELY WRONG results? (Alternative join approaches are OK)
2. Are aggregations CLEARLY WRONG for the question? (e.g., COUNT when explicitly asking for SUM) (Variations are OK)

Only mark as invalid if there are SERIOUS problems. If the SQL could reasonably work, mark it as VALID.

Provide your analysis."""


def create_entity_extraction_prompt(question: str, custom_prompts: str = "") -> str:
    if custom_prompts:
        return f"""Extract database entities from this question.

Domain-specific rules:
{custom_prompts}

Question: {question}

Return:
1) required_entity_name: all database concepts from the question
   (table names, column names, relationships). Ignore values, dates, numbers.

2) item_search_queries: 2-4 search phrases (1-3 words each) to find relevant database tables
   and columns. Include exact table/column names from the domain rules above
   that are relevant, plus rephrased angles from the question.
"""

    return f"""Extract database entities from this question.

Question: {question}

Return:
- required_entity_name: all database concepts from the question
   (table names, column names, relationships). Ignore values, dates, numbers.
"""


TABLE_RELEVANCE_FILTER_PROMPT = """You are a database schema expert.
Given a user's question and a list of candidate tables, decide which tables
are actually needed to answer the question.

Rules:
- Only remove tables you are confident are NOT needed in the SQL query.
- If table A must be joined through table B to reach table C, do NOT
  remove any table in the join chain (A, B, or C).
- When in doubt, do NOT remove — it is safer to include an extra table
  than to remove a necessary one.

{domain_rules}
User's question:
{question}

Candidate tables:
{tables_summary}

Provide brief reasoning (1-2 sentences) then return the names of tables that can be safely REMOVED.
Only remove a table if you are confident it is not needed. When in doubt, do NOT remove."""
