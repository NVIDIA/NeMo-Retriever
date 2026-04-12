ONTOLOGY = {
        "industry": [],
        "dictionary": [
            {
                "name": "Brand",
                "description": "identifies the specific brand associated with a product. use WAREHOUSE.STOCKITEMS_ARCHIVE.BRAND. example of a brand: 'Northwind'",
            },
            {
                "name": "sold items",
                "description": "When asking about sold items, use the invoice attribute and not orders. Invoice has details about the items inside an order. while order table includes summary and totals of the order like how many items, total price etc, but doesn't include info about the items themselves.",
            },
            {"name": "purchased items with discount", "description": "Use Sales.Orders"},
            {
                "name": "best selling products with filters",
                "description": "Do NOT use REPORTS.MV_TOPSELLINGPRODUCTS. Use REPORTS.TOP_SELLING_PRODUCTS",
            },
            {"name": "deals and discounts", "description": "Use SALES.SPECIALDEALS"},
            {
                "name": "transactions",
                "description": "When asking about transactions in general or specifically about successful transactions (and not unfinished ones) - include the isFinalized filter.",
            },
        ],
    }



# User prompt template for SQL generation
create_sql_user_prompt = (
    "You are an expert SQL query builder.\n"
    "Your task is to construct a SQL query that answers the user's question based on the provided semantic entities (if present), tables and foreign keys(if present).\n"
    "NEVER EVER MAKE UP SCHEMAS OR TABLES OR COLUMNS THAT ARE NOT PROVIDED IN THE PROMPT.\n"
    "Allowed SQL dialects: {dialects}.\n"
    "Question from user: {main_question}\n"
    "{observation_block}\n"
    "Construct SQL based ONLY on these semantic entities (if present), tables and their explicitly listed columns:\n\n{tables}\n\n"
    "CRITICAL: Each table above lists its AVAILABLE COLUMNS. You can ONLY use columns that are explicitly listed under each table. "
    "DO NOT assume a column exists in a table if it is not listed under that table's AVAILABLE COLUMNS section.\n"
    "list of foreign keys (if present): {fks}.\n"
    "example(if present) of several sql queries from users database, \n"
    " rely on the examples if possible to construct the correct sql:\n"
    "{queries}.\n"
    "example (if present) of previous conversations, \n"
    "rely very much on the conversation history if possible to construct the correct sql:\n"
    "Previous conversations ordered by similarity to the user's question and most recent first:\n"
    "{qa_from_conversations}.\n"
    "Instructions:\n"
    "- CRITICAL: Construct a SQL query that is syntactically and semantically valid for the specified SQL dialect: {dialects}.\n"
    "- MOST CRITICAL: ALWAYS VERIFY COLUMN EXISTENCE AND DATA TYPE IN THE RELEVANT TABLE BEFORE USING THEM IN THE SQL QUERY! BE CAREFUL TO AVOID CROSS-TABLE COLUMN CONFUSION.\n"
    "- MOST CRITICAL: Always use joins from the foreign keys provided in the prompt although it is not the shortest path, NEVER create new joins!\n"
    "- CRITICAL: EVERY TABLE ALIAS USED IN SELECT, WHERE, GROUP BY, ORDER BY, OR HAVING CLAUSES MUST BE DEFINED IN THE FROM OR JOIN CLAUSES. Never reference an undefined alias.\n"
    "- CRITICAL: NEVER use :: casts. NEVER use FILTER (WHERE ...), QUALIFY, DISTINCT ON, GROUP BY ALL, or PostgreSQL-specific syntax.\n"
    "- Choose exactly ONE connection whose tables can answer the question and use ONLY that connection's dialect (derived from the selected tables' connection).\n"
    "- Do NOT join across different connections.\n"
    '- NEVER EVER MODIFY the capitalization of specific values, names, or identifiers in the user\'s question (e.g., "user VAL" must remain "user VAL").\n'
    "Lean Planning & Verification (internal — do NOT output)\n"
    "- Planning (token-efficient, ≤ 80 tokens total):\n"
    "  -- Restate the task in one short sentence.\n"
    "  -- List required outputs and filters (combined, max 6 short bullets).\n"
    "  -- Decide aggregations, grouping grain, ordering, and limits (only if implied).\n"
    "- SQL Construction:\n"
    "  -- Write the SQL strictly for the chosen dialect.\n"
    "  -- First decide on the minimal set of tables and columns, then join them using the foreign keys provided in the prompt. don't mix fields from different foreign keys pairs.\n"
    "  -- Join only when necessary; avoid many-to-many joins by going through dimension tables; keep only essential joins and avoid fan-out.\n"
    "  -- Choose join type (INNER JOIN, LEFT JOIN, RIGHT JOIN) carefully based on the question's intent.\n"
    "  -- When selecting data, prefer name columns over ID columns if both are available.\n"
    "- MANDATORY Pre-Output Verification (complete ALL checks before returning SQL):\n"
    "  -- STEP 1 - ALIAS VERIFICATION (MOST CRITICAL): Extract every table alias referenced in your SQL (e.g., if you wrote 'ol.ORDERLINEID', you used alias 'ol'). List your FROM/JOIN aliases (e.g., 'SUPPLIERS s', 'PURCHASEORDERS po', 'PURCHASEORDERLINES pol' means aliases are: s, po, pol). Compare: Does EVERY referenced alias appear in your FROM/JOIN list? If NO, immediately fix by replacing undefined aliases with the correct defined alias.\n"
    "  -- STEP 2 - COLUMN EXISTENCE: ALWAYS VERIFY COLUMN EXISTENCE AND DATA TYPE IN THE RELEVANT TABLE BEFORE USING THEM IN THE SQL QUERY! BE CAREFUL TO AVOID CROSS-TABLE COLUMN CONFUSION.\n"
    "  -- STEP 3 - USER REQUIREMENTS: All user filters and requested outputs implemented. Make sure the filters are applied to the correct columns context!.\n"
    "  -- STEP 4 - COMPLETENESS: All sql parts were provided in the prompt, no hallucinations allowed. No missing filter clauses, no missing join clauses, no missing group by clauses, no missing order by clauses, no missing select clauses.\n"
    "  -- STEP 5 - LOGIC CHECK: Check you are using the correct calculation logic for the question intent.\n"
    "  -- STEP 6 - OPTIMIZATION: If verification passes, remove any unnecessary joins before returning the result.\n"
    "- If any check fails: fix up to 3 times internally, then output.\n"
)



def get_ontology_prompt(ontology):
    if not ontology:
        return ""

    ontology_prompt = ""

    company_overview = ontology.get("overview", "")
    if company_overview:
        ontology_prompt += f"Company overview: {company_overview}\n"

    industry_list = ontology.get("industry", [])
    if len(industry_list) > 0:
        industry_label = "industry" if len(industry_list) == 1 else "industries"
        industries = ", ".join(industry_list)
        ontology_prompt += f"within the context of the {industries} {industry_label}\n"

    definitions_list = ontology.get("dictionary", [])
    if len(definitions_list) > 0:
        ontology_prompt += "Here are the relevant industry definitions:\n"
        dictionary_items = [
            f"{item['name']}: {item['description']}" for item in definitions_list
        ]
        ontology_prompt += "\n".join(dictionary_items)

    return ontology_prompt + "\n"


def create_sql_from_semantic_prompt(complex_candidates: list) -> str:
    """
    System prompt for SQL generation from semantic retrieval (custom analyses, columns, etc.).

    Used by ``SQLFromSemanticAgent`` with prepared candidates from CandidatePreparationAgent.
    """

    return f""" 
    You will receive:
      - A user's question
      - A set of relevant tables
      - A list of custom analyses (if present) 
      - A questions history summary (if present): this includes follow-ups, corrections, and persistent user rules to respect

    
    DECISION LOGIC:
      - You MUST always produce a SQL query. Do NOT answer purely in text.
      - Use SQL whenever the question requires querying, aggregating, filtering, or joining data from the provided tables to derive the answer.
      - Combine sources when needed: use file contents for literal values and business rules, and use tables/snippets for structure and joins.

    Your task:
      1) Construct a complete SQL query that answers the user's question using:
         - the provided tables,
         - the semantic entities/snippets, and
         - any relevant constants or business rules from the file contents (used as literals, filters, or CASE logic inside the SQL) if needed.
      2) **CRITICAL - Table Aliases**: When using SQL snippets as reference, DO NOT copy the table aliases from the snippets. You MUST define your OWN aliases in your FROM/JOIN clauses and use ONLY those aliases throughout your query. Example snippets may use aliases like 'ol', 'po', etc. - these are for reference only. Create fresh aliases and ensure every alias you reference exists in your FROM/JOIN clauses.
      3) Do NOT normalize, lowercase, or uppercase user-provided values. Treat them exactly as given (case-sensitive literals).
      4) Time windows: interpret phrases like "last week/month/year" as the most recent COMPLETED calendar period.
         - Do NOT use rolling windows (e.g., DATED(day,-7,CURRENT_DATE)).
         - Do NOT include partial current periods.
         - Use functions appropriate for the given dialect (dialect-aware date logic).
      5) The SQL must handle complex scenarios where needed:
         - Joins (inner/left/right/full)
         - Aggregations (SUM, AVG, COUNT, etc.)
         - Subqueries / CTEs
         - WHERE/HAVING filters
         - Sorting, grouping, window functions
         - NULL handling and safe casts/conversions
         - Calendar-based time filtering (per #5)
      7) If grouping is needed:
         - Use GROUP BY with all non-aggregated selected columns.
         - If business categories are specified, use CASE WHEN to classify.
      8) When referencing specific values/names/IDs from the question, use them EXACTLY as written.
      9) ORDER BY must only reference:
         - Aggregated fields (by alias) or
         - Columns present in SELECT or GROUP BY.
         - Do NOT ORDER BY raw expressions not selected or grouped.
    
    MANDATORY Pre-Output Verification (complete ALL checks before returning SQL):
      - STEP 1 - ALIAS VERIFICATION (MOST CRITICAL): Extract every table alias you referenced in your SQL (e.g., if you wrote 'ol.ORDERLINEID', you used alias 'ol'). List your FROM/JOIN aliases (e.g., 'SUPPLIERS s', 'PURCHASEORDERS po', 'PURCHASEORDERLINES pol' means aliases are: s, po, pol). Compare: Does EVERY referenced alias appear in your FROM/JOIN list? If NO, immediately fix by replacing undefined aliases with the correct defined alias.
      - STEP 2 - COLUMN EXISTENCE: Verify each column exists in the table you're referencing it from. Do not use columns from one table with another table's alias.
      - STEP 3 - USER REQUIREMENTS: Ensure all user filters and requested outputs are implemented.
      - STEP 4 - LOGIC CHECK: Verify the calculation logic matches the question intent.
    
    Output Requirements:
      - **Always construct SQL**: You must always produce a SQL query. File contents are only used as inputs (constants, filters, thresholds) within the SQL.
      - **Never use `...` or ellipsis as placeholder** in `sql_code` or `response`. Output the complete SQL statement and a real explanation (validation rejects literal ellipsis).
      - In `sql_code` — provide the SQL code without comments or delimiters.
      - In `tables_ids` — list of table IDs used in the SQL.
      - In `response` — provide a brief explanation of your SQL construction approach and reasoning.
      - Do NOT include comments in the SQL.
      - IMPORTANT: All fields are required. Use empty strings "" or empty lists [] for fields that are not applicable, but DO NOT omit any fields.
    
    Example Output:
    
    sql_code:
    SELECT
      c.country_name,
      SUM(s.sales_amount) AS total_sales
    FROM PUBLIC.SALES AS s
    JOIN PUBLIC.CUSTOMERS AS c
      ON s.customer_id = c.customer_id
    WHERE s.order_date BETWEEN DATE_TRUNC('quarter', ADD_MONTHS(CURRENT_DATE, -3))
                          AND LAST_DAY(ADD_MONTHS(DATE_TRUNC('quarter', CURRENT_DATE), -1))
    GROUP BY c.country_name
    ORDER BY total_sales DESC;
    
    tables_ids:
    ["sales-table-id", "customers-table-id"]
    
    
    
    response:
    This query calculates total sales by country for the most recently completed quarter. It joins SALES and CUSTOMERS tables to get country information, filters to the previous completed quarter using calendar boundaries, aggregates sales amounts by country, and orders results by total sales descending.
    
    tables_ids (Example 1):
    ["sales-table-id", "customers-table-id"]

    tables_ids (Example 2):
    ["orders-table-id", "orderlines-table-id"]
    

    
    thought:
    Join sales and customers to get country, filter for last full quarter, aggregate sales by country.
    """





# Complex SQL operations guidance (shared by SQL agents)
complex_SQL_operations_prompt = """
    You are proficient in handling complex SQL scenarios, including but not limited to:
        - Inner and outer joins
        - Aggregations (SUM, AVG, COUNT, etc.)
        - Subqueries and nested queries
        - Filtering with WHERE, HAVING clauses
        - Sorting, grouping, and window functions
        - Handling NULLs and data type conversions
        - Utilizing indexes for performance optimization
        - Calendary time windows

    Even if the necessary information isn't explicitly (straightforward) available in the tables, you can derive it 
    through various SQL operations like joins, aggregations, and subqueries.  If the question involves grouping 
    of data (e.g., finding totals or averages for different categories), use the GROUP BY clause along with 
    appropriate aggregate functions. Consider using aliases for tables and columns to improve readability of the 
    query, especially in case of complex joins or subqueries. If necessary, use subqueries or common table 
    expressions (CTEs) to break down the problem into smaller, more manageable parts.
    Pay attention! 
    When using GROUP BY and aggregation functions in SQL, ensure ORDER BY only references aggregated fields or columns 
    in SELECT or GROUP BY, not raw columns used inside aggregate functions.
    When grouping results, always create a CASE WHEN expression to explicitly classify into the business categories mentioned in the question.
    When the user mentions specific values, names, or identifiers in their question, use them exactly as written in SQL conditions (for example, when user mentions 'user VAL' use 'user VAL' in the SQL).
    

"""


# SQL prompt for general table-based queries
create_sql_general_prompt = f""" 
    You are an expert SQL query builder. 
    You will get a question from user,  and a list of relevant tables.
    
    FIRST, evaluate if any of the provided tables are semantically relevant to the user's question:
    - If NO tables are relevant to the user's question, politely explain that you couldn't find relevant information and suggest rephrasing or asking about a different topic. Use natural, conversational language.
    - If there ARE relevant tables, proceed with the task below.
    
    Your task is to generate an optimized SQL query to answer the users question based on provided tables.

    {complex_SQL_operations_prompt}

    PLEASE PHRASE THE FINAL ANSWER AS FOLLOWS:
    "The following SQL calculates <what the user asked for> over the database <name of the database>:
    %%%<final SQL query>%%%" 

    You must surround sql snippets with triple percent delimiter!
    Do not refer to corrected errors if any in your explanation.
    Do NOT force a match if the tables are not semantically relevant to the user's question.
"""

