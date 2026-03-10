import logging
import re

# TODO how to do the connection
from nemo_retriever.relational_db.infra.Neo4jConnection import get_neo4j_conn


from nemo_retriever.relational_db.description_suggestion.system_prompts import (
    term_system_prompt,
    attribute_system_prompt,
    metric_system_prompt,
    analysis_system_prompt,
    analysis_name_system_prompt,
    sql_system_prompt,
    system_intro_prompt,
    table_system_prompt,
    schema_system_prompt,
    column_system_prompt,
)

logger = logging.getLogger("description suggestion")
conn = get_neo4j_conn()

# TODO where is ontology coming from?
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


def get_final_prompt(prompt, system_prompt, ontology):
    ontology_prompt = get_ontology_prompt(ontology)
    system_prompt = system_intro_prompt + ontology_prompt + system_prompt
    return prompt, system_prompt


def create_table_prompt(data, ontology):
    if data is None or not data.keys() >= {"schema_name", "table_name", "columns"}:
        return None, None

    prompt = f"The table name: '{data['table_name']}', schema name: '{data['schema_name']}', columns: {data['columns']}."
    prompt, system_prompt = get_final_prompt(prompt, table_system_prompt, ontology)
    return prompt, system_prompt


def create_schema_prompt(data, ontology):
    if data is None or not data.keys() >= {"schema_name", "db_name", "tables"}:
        return None, None

    prompt = f"The schema name: '{data['schema_name']}', database name: '{data['db_name']}', tables: {data['tables']}."
    prompt, system_prompt = get_final_prompt(prompt, schema_system_prompt, ontology)
    return prompt, system_prompt


def create_column_prompt(data, ontology):
    if data is None or not data.keys() >= {"table_name", "column_name", "columns"}:
        return None, None

    prompt = f"The specific column name: '{data['column_name']}', table name: '{data['table_name']}', a list of other column names within the same table: {data['columns']}."
    prompt, system_prompt = get_final_prompt(prompt, column_system_prompt, ontology)
    return prompt, system_prompt


def create_term_prompt(data, ontology):
    if data is None or not data.keys() >= {"name"}:
        return None, None

    prompt = f"The term: '{data['name']}'."
    prompt, system_prompt = get_final_prompt(prompt, term_system_prompt, ontology)
    return prompt, system_prompt


def create_attribute_prompt(data, ontology):
    if data is None or not data.keys() >= {"term_name", "attribute_name"}:
        return None, None

    prompt = f"The attribute: '{data['attribute_name']}', the term that contains the attribute: '{data['term_name']}'."
    prompt, system_prompt = get_final_prompt(prompt, attribute_system_prompt, ontology)
    return prompt, system_prompt


def create_metric_prompt(data, ontology):
    if data is None or not data.keys() >= {"formula", "name"}:
        return None, None

    prompt = (
        f"The metric name: '{data['name']}', the metric formula: '{data['formula']}'."
    )
    prompt, system_prompt = get_final_prompt(prompt, metric_system_prompt, ontology)
    return prompt, system_prompt


def create_analysis_prompt(data, ontology):
    if data is None or not data.keys() >= {"query_text", "name"}:
        return None, None

    prompt = (
        f"the analysis name: '{data['name']}', the SQL query: '{data['query_text']}'."
    )
    prompt, system_prompt = get_final_prompt(prompt, analysis_system_prompt, ontology)
    return prompt, system_prompt


def create_analysis_name_prompt(data, ontology):
    if data is None or not data.keys() >= {"query_text"}:
        return None, None

    sql = remove_sql_comments(data["query_text"])
    prompt = f"the SQL query: '{sql}'."
    prompt, system_prompt = get_final_prompt(
        prompt, analysis_name_system_prompt, ontology
    )
    return prompt, system_prompt


# This was used in population, should be used if we will have for single query
def create_query_prompt(data, ontology):
    if data is None or not data.keys() >= {"query_text"}:
        return None, None

    sql = remove_sql_comments(data["query_text"])
    prompt = f"the SQL query: '{sql}'."
    prompt, system_prompt = get_final_prompt(prompt, sql_system_prompt, ontology)

    return prompt, system_prompt





def fix_response(descriptions: str):
    return descriptions.replace("\n", "").replace("'", "").replace('"', "")


def remove_sql_comments(sql_query: str):
    # Remove single-line comments
    sql_query = re.sub(r"--.*$", "", sql_query, flags=re.MULTILINE)
    sql_query = re.sub(r"//.*$", "", sql_query, flags=re.MULTILINE)

    # Remove multi-line comments
    sql_query = re.sub(r"/\*.*?\*/", "", sql_query, flags=re.DOTALL)

    return sql_query

