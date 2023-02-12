from lfqa import *
def insert_to_database_test():
    DBNAME = os.getenv("DBNAME", "postgres")
    HOST = os.getenv("HOST", "db")
    PORT = os.getenv("PORT", "5432")
    USER = os.getenv("USER", "postgres")
    PWD = os.getenv("PASSWORD", "12345")

    try:
        connection = psycopg2.connect(dbname=DBNAME, user=USER, host=HOST, port=PORT, password=PWD)
        connection.autocommit = True
        cursor = connection.cursor()
        query = f"""
                    CREATE DATABASE kiki40;
                """
        cursor.execute(query)
        if connection:
            cursor.close()
            connection.close()

    except (Exception, psycopg2.Error) as error:
        print("Failed insert record from database {}".format(error))


import psycopg2

# connection establishment
conn = psycopg2.connect(
    database="postgres",
    user='postgres',
    password='12345',
)

conn.autocommit = True

# Creating a cursor object
cursor = conn.cursor()

# query to create a database
sql = ''' CREATE extension vector ''';

# executing above query
cursor.execute(sql)
print("Database has been created successfully !!");

# Closing the connection
conn.close()