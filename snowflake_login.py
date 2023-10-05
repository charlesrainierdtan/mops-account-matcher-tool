import snowflake.connector
from snowflake.connector import DictCursor
import os


class connect:
    connection = None
    
    def __init__(self):
        user = os.getenv('SNOWFLAKE_USER')
        account = 'genesysinc'
        warehouse = 'MARKETINGOPERATIONS'
        password = os.getenv('SNOWFLAKE_PASSWORD')
        # authenticator = 'externalbrowser'
        
        self.connection = snowflake.connector.connect(
            user=user,
            account=account,
            warehouse=warehouse,
            password=password
		)
    
    def execute(self, sql):
        return self.connection.cursor().execute(sql)
    
    def execute_dict(self, sql):
        return self.connection.cursor(DictCursor).execute(sql).fetchall()
    
    def executeFromFile(self, sqlFile):
        with open(sqlFile, 'r', encoding='utf-8') as f:
            sql = f.read()
        return self.execute_dict(sql)