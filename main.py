import re
import psycopg2
from sql import get_predicted_querys

if __name__ == '__main__':
    # Input: tables, SQL_list
    # Output: list_db, actua_db, predicted_query()
    # ==================================================================================================================
    # Step1 Get information about tables in the database
    tables = dict()  # tables = {"table1": [tuple_num, [column_num, [c1_name, c2_name, ...]]], ...}
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="postgres",
        user="postgres",
        password="123456"
    )
    cur = conn.cursor()
    cur.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'")
    table_num = 0
    for table in cur.fetchall():
        table_num += 1
        table_name = table[0]
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        tuple_num = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM information_schema.columns WHERE table_name='{table_name}'")
        column_num = cur.fetchone()[0]
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name='{table_name}'")
        columns = cur.fetchall()
        column_names = [column[0] for column in columns]
        tables[table_name] = [tuple_num, [column_num, column_names]]
    cur.close()
    conn.close()
    # ==================================================================================================================
    # Step2 Obtain list_db, actua_db, predicted_query
    list_db = []
    actua_db = []
    for name, information in tables.items():
        list_db.append([10, information[1][0]])
        actua_db.append(information[0])
    # read .sql
    # path = r'D:\Data\TPC_DS\sql'
    sqls = dict()
    folder_path = "D:/Data/TPC_DS/sql/"
    for i in range(1, 100):
        file_path = folder_path + "query" + str(i) + ".sql"
        with open(file_path, 'r') as file:
            sql_name = "sql" + str(i)
            lines = file.readlines()
            lines = lines[1:-2]
            merged_lines = ''.join(lines).replace('\n', ' ')
            merged_lines = re.sub(r'\s+', ' ', merged_lines)
            sqls[sql_name] = [merged_lines, 1 / 99]
    predicted_query = get_predicted_querys(sqls, tables)

