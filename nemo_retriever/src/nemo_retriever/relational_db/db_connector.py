from connectors.connector import Connector
import json


class DBConnector(Connector):
    def __init__(self, connection):
        super().__init__(connection)

    @property
    def databases(self):
        return self.__databases

    @databases.setter
    def databases(self, value):
        self.__databases = value

    @property
    def pull_info(self):
        return self.__pull_info

    @pull_info.setter
    def pull_info(self, value):
        if isinstance(value, str):
            self.__pull_info = json.loads(value)
        else:
            self.__pull_info = value
