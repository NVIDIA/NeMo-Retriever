import json
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
from connectors.connector import DBConnector
from connectors.utils import group_by_queries


class BigQuery(DBConnector):
    def __init__(self, connection):
        super().__init__(connection)
        if "keyfile" in self.connection_properties:
            json_keyfile_string = self.connection_properties["keyfile"]
            json_keyfile = json.loads(json_keyfile_string)
            credentials = service_account.Credentials.from_service_account_info(
                json_keyfile
            )
            self.client = bigquery.Client(credentials=credentials)

    def _query(self, query):
        job = self.client.query(query)
        return job.result().to_dataframe()

    def get_regions_of_project(self, project):
        datasets = list(self.client.list_datasets(project=project))
        regions = set()
        for ds in datasets:
            # This property is not exposed :( risk...
            regions.add(ds._properties["location"])
        return list(regions)

    def get_datasets_of_project(self, project):
        datasets = list(self.client.list_datasets(project=project))
        return [dataset.dataset_id for dataset in datasets]

    def test(self):
        # Project in BQ is database in illumex/snowflake
        # Dataset in BQ is schema in illumex/snowflake
        to_return = []
        projects = list(self.client.list_projects())
        projects = [project.project_id for project in projects]
        for project in projects:
            datasets = self.get_datasets_of_project(project)
            to_return.append({"db_name": project, "schemas": datasets})
        return to_return

    def get_schemas(self):
        def get_columns():
            def get(db):
                datasets = self.get_datasets_of_project(db)

                def get_per_dataset(ds):
                    q = f"""
                    SELECT table_catalog as `database`, table_schema as `schema`, table_name as `table_name`, column_name as `column_name`, ordinal_position as `ordinal_position`, data_type as `data_type`, is_nullable as `is_nullable`
                    FROM `{db}.{ds}.INFORMATION_SCHEMA.COLUMNS`;
                    """
                    schema = self._query(q)
                    return schema

                columns_per_ds = map(get_per_dataset, datasets)
                columns_per_ds = pd.concat(columns_per_ds, ignore_index=True)
                return columns_per_ds

            schemas = map(get, self.databases)
            return pd.concat(schemas, ignore_index=True)

        def get_tables():
            def get(db):
                datasets = self.get_datasets_of_project(db)

                def get_per_dataset(ds):
                    q = f"""
                    SELECT table_catalog as `database`, table_schema as `schema`, table_name as `table_name`, table_type as `table_type`, creation_time as `created`
                    FROM `{db}.{ds}.INFORMATION_SCHEMA.TABLES`;
                    """
                    schema = self._query(q)
                    return schema

                tables_per_ds = map(get_per_dataset, datasets)
                tables_per_ds = pd.concat(tables_per_ds, ignore_index=True)
                return tables_per_ds

            schemas = map(get, self.databases)
            return pd.concat(schemas, ignore_index=True)

        return get_tables(), get_columns()

    def get_queries(self, data_interval_start, data_interval_end):
        def get(db):
            regions = self.get_regions_of_project(db)

            def get_queries_per_region(region):
                q = f"""
                SELECT end_time, query as query_text
                FROM `{db}.region-{region}.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
                WHERE creation_time >= TIMESTAMP('{str(data_interval_start)}') AND end_time <= TIMESTAMP('{str(data_interval_end)}')
                AND statement_type != 'CREATE' AND statement_type != 'DROP' AND statement_type != 'ALTER'
                AND state = 'DONE' AND error_result IS NULL
                AND NOT REGEXP_CONTAINS(query, r'(?i)information_schema')
                ORDER BY end_time DESC;
                """
                return self._query(q)

            queries_per_region = map(get_queries_per_region, regions)
            queries_per_region = pd.concat(queries_per_region, ignore_index=True)
            return queries_per_region

        queries = map(get, self.databases)
        queries = pd.concat(queries, ignore_index=True)
        queries = group_by_queries(queries)
        return queries

    def get_views(self):
        def get(db):
            datasets = self.get_datasets_of_project(db)

            def get_per_dataset(ds):
                q = f"""
                SELECT table_catalog as `database`, table_schema as `schema`, table_name as `table_name`, view_definition as `view_definition`
                FROM `{db}.{ds}.INFORMATION_SCHEMA.VIEWS`;
                """
                views = self._query(q)
                return views

            views_per_ds = map(get_per_dataset, datasets)
            views_per_ds = pd.concat(views_per_ds, ignore_index=True)
            return views_per_ds

        views = map(get, self.databases)
        views = pd.concat(views, ignore_index=True)
        return views
