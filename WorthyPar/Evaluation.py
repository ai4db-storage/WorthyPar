import psycopg2
import random
from WorthyPar.Extraction import get_queries_name, get_queries, get_table_size
Partition = {'WorthyPar': 'p',
             'DRL-H': 'p0',
             'Horizontal-H': 'p1',
             'Horizontal-V': 'p2'}


def generate_workload(dbname, workload_size):
    queries = get_queries_name(dbname)
    workload = []
    for i in range(workload_size):
        index = random.randint(0, len(queries)-1)
        workload.append(queries[index])
    return workload


def generate_query_plans(dbname, pname, workload, Queries):
    queries = get_queries(dbname)
    sql_list = []
    size_list = []
    if pname == 'origin':
        for query_name, query in Queries.items():
            sql_list.append(queries[query_name])
            size = 0
            for table_name in query['tables']:
                size += get_table_size(dbname, table_name)
            size_list.append(size)
    else:
        for query_name, query in Queries.items():
            sql = queries[query_name]
            size = 0
            for table_name in query['tables']:
                partition_name = table_name + "_" + query_name + "_" + Partition[pname]
                sql = sql.replace(" "+table_name+" ", " "+partition_name+" ")
                size += get_table_size(dbname, partition_name)
            sql_list.append(sql)
            size_list.append(size)

    explain = []
    scan_size = 0
    for q in workload:
        index = list(queries.keys()).index(q)
        explain.append(sql_list[index])
        scan_size += size_list[index]
    # print(explain)
    return explain, scan_size


def execution_query_plans(dbname, query_plans, repeat, user, password, host="localhost",port=5432):
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=dbname,
        user=user,
        password=password
    )
    cur = conn.cursor()
    time = []
    for i in range(repeat):
        temp = 0
        for q in query_plans:
            cur.execute(f"EXPLAIN ANALYZE {q}")
            result = cur.fetchall()
            for row in result:
                if "Execution Time" in row[0]:
                    execution_time = row[0].split(" ")[-2]
                    break
            temp += float(execution_time)
        time.append(temp)
        print(i, temp)
    cur.close()
    conn.close()
    runtime = sum(time)/len(time)
    return runtime

