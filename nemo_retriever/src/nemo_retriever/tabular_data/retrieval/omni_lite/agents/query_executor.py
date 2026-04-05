import os
from typing import Optional, Union
import requests
import asyncio
import logging

logger = logging.getLogger("query_execution")



class QueryResponse:
    def __init__(self, result: list[str], sliced: bool, error: Optional[str] = None):
        self.result = result
        self.sliced = sliced
        self.error = error


async def run_query(
    account: str,
    sql: str,
    connection_id: str,
    db_name: str,
    source: str,
    tag: str = None,
) -> Optional[Union[QueryResponse, dict]]:
    connections_url = os.environ.get("LMX_CONNECTIONS_SERVICE_URL")
    url = f"{connections_url}/connections/query?account={account}"
    payload = {
        "connection_id": connection_id,
        "db_name": db_name,
        "query": sql,
        "format_response": source not in [AgentSource.EXTENSION], # TODO ??
        "tag": tag,
    }
    headers = {"Content-Type": "application/json"}

    def make_request():
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Connections query error!: {e}")
            return {"result": None, "sliced": False, "error": str(e)}

    response_data = await asyncio.to_thread(make_request)
    try:
        return QueryResponse(
            result=response_data.get("result", None),
            sliced=response_data.get("sliced", False),
            error=response_data.get("error"),
        )
    except (KeyError, TypeError) as e:
        logger.error(
            f"Error run query: {str(e)}\nAccount: {account}\nSQL: {sql}\nConnection ID: {connection_id}\nDB Name: {db_name}\nSource: {source}"
        )
        return QueryResponse(result=None, sliced=False, error=str(e))
