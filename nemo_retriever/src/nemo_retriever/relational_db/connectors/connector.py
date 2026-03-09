class Connector:
    def __init__(self, connection):
        self.__connection_properties = connection

    @property
    def connection_properties(self):
        return self.__connection_properties

    # This should be ovrrided
    def test(self):
        return None
