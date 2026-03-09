from infra.Neo4jConnection import get_neo4j_conn
import logging

logger = logging.getLogger("terms_dal.py")

conn = get_neo4j_conn()


def bt_exists_in_graph(bus_term: str, account_id: str):
    query_bt_exists = f""" MATCH (n {{name:"{bus_term}", account_id:"{account_id}"}}) 
                            WHERE (n:term) 
                            RETURN n
                        """
    result_bt = conn.query_read_only(
        query=query_bt_exists, parameters={"account_id": account_id}
    )
    if len(result_bt) > 0:
        return True
    return False


def update_attrs_bts(account_id, att_id=None):
    if att_id:
        query = """MATCH (d:attribute {account_id: $account_id, id:$att_id})"""
    else:
        query = """MATCH (d:attribute {account_id: $account_id}) """

    # invalid_time: if invalid set to true and time was not already set -> set invalid_time, else leave invalid_time unchanged
    query += """ 
                OPTIONAL MATCH (d)-[:attr_of]->(c)
                call apoc.case([c is null,
                                'return false as res',
                                c:column,
                                'return coalesce(c.deleted, false) as res',
                                c:alias or c:sql,
                                'optional match p=(d)-[:attr_of]->(c)-[:SQL*]->(c2:column{deleted:true}) return p is not null as res'
                                ],
                                'return false as res',
                                {c:c})
                yield value
                WITH d, collect(value.res) as att_invalid
                SET d.invalid = reduce(del_res = false, del_a IN att_invalid | del_res OR del_a)
                SET d.certified = case when d.invalid=true then false else d.certified end
                SET d.invalid_time = case when d.invalid=true and d.invalid_time is null 
                    then datetime.realtime() else d.invalid_time end
            """
    conn.query_write(
        query=query, parameters={"account_id": account_id, "att_id": att_id}
    )

    query = """ MATCH (bt:term {account_id: $account_id}) """

    if att_id:
        query += """ MATCH (bt)-[:term_of]->(a:attribute{id:$att_id}) """
    else:
        query += """ MATCH (bt)-[:term_of]->(a:attribute) """
    query += """
             WITH collect(coalesce(a.invalid, false)) as att_invalid, bt
             SET bt.invalid = reduce(del_res = false, del_a IN att_invalid | del_res OR del_a)
             SET bt.invalid_time = case when bt.invalid=true and bt.invalid_time is null 
                    then datetime.realtime() else bt.invalid_time end
             return collect(properties(bt)) as bt_props
            """
    conn.query_write(
        query=query, parameters={"account_id": account_id, "att_id": att_id}
    )


def get_bt_name_by_attr_id(attr_id, account_id):
    query = """MATCH (t:term)-[:term_of]->(a:attribute{id:$attr_id, account_id: $account_id})
               RETURN t.name as term_name
            """
    result = conn.query_read_only(
        query=query, parameters={"account_id": account_id, "attr_id": attr_id}
    )
    if len(result) > 0:
        return result[0]["term_name"]
    else:
        raise "Something not ok in finding term by attribute id."


def get_term_id_by_name(account_id, term_name):
    query = """match(t:term{account_id:$account_id, name: $term_name}) return t.id as term_id"""
    return conn.query_read_only(
        query=query, parameters={"account_id": account_id, "term_name": term_name}
    )


def get_attribute_id_by_names(account_id, term_name, attr_name):
    query = """match(t:term{account_id:$account_id, name:$term_name})-[:term_of]-(a:attribute{name:$attr_name}) return a.id as attr_id"""
    res = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "term_name": term_name,
            "attr_name": attr_name,
        },
    )
    if len(res) > 0:
        return res[0]["attr_id"]
    return "ignore"
