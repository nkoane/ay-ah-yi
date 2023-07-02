
import psycopg2
import os
from dotenv import load_dotenv


def get_db_connection():
    import psycopg2

    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

    return conn


''''
conn = get_db_connection();


# Do some database operations here
print("Connected to database", conn)
# Create a cursor object
cur = conn.cursor()

# Execute a query
cur.execute("SELECT * FROM documents")

# Fetch the results
results = cur.fetchall()

# Close the cursor and connection
cur.close()
conn.close()

'''
