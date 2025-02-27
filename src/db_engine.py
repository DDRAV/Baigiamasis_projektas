"""
Database file to connect, execute commands, and close connection.
"""

import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_batch

# Load environment variables
load_dotenv()

class DBEngine:
    def __init__(self):
        self.connection = self.connect()
        self.cursor = self.connection.cursor()

    @staticmethod
    def connect():
        """
        Connect to the database using credentials from the .env file.
        :return: Database connection object.
        """
        try:
            connection = psycopg2.connect(
                dbname=os.getenv('DATABASE_NAME'),
                user=os.getenv('DB_USERNAME'),
                password=os.getenv('PASSWORD'),
                host=os.getenv('HOST'),
                port=os.getenv('PORT')
            )
            print("PostgreSQL connection is opened")
            return connection
        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL:", error)
            return None

    def execute_batch(self, query, data_list):
        """
        Execute a batch of SQL queries efficiently using execute_batch.
        :param query: SQL query with placeholders.
        :param data_list: List of tuples containing the data to insert.
        :return: None.
        """
        try:
            execute_batch(self.cursor, query, data_list)
            self.connection.commit()
            print("Batch update successful!")
        except (Exception, psycopg2.Error) as error:
            print("Error while executing batch update:", error)
            self.connection.rollback()

    def execute_sql(self, query, params=None):
        """
        Execute an SQL query.
        :param query: The SQL command to execute.
        :param params: Optional parameters for the query.
        :return: Query result as a tuple.
        """
        try:
            self.cursor.execute(query, params)
            self.connection.commit()
            return self.cursor.fetchall()
        except (Exception, psycopg2.Error) as error:
            print("Error while executing query:", error)
            self.connection.rollback()
            return None

    def disconnect(self):
        """
        Close the database connection.
        :return: None.
        """
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
                print("PostgreSQL connection is closed")
        except (Exception, psycopg2.Error) as error:
            print("Error while closing PostgreSQL connection:", error)

    def __del__(self):
        self.disconnect()

if __name__ == "__main__":
    db = DBEngine()
    if db.connection:
        result = db.execute_sql(
            # Example queries (commented out):
            # "INSERT INTO test (lyrics, chords) VALUES ('So close no matter how far, couldnt be much more from the heart', 'Em > D > C > Em > D > C');"
            # "SELECT * FROM test;"
            # "DELETE FROM test;"
        )
        print(f"{result}")
        print("Command executed")
