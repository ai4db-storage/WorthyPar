# Extraction Module

import psycopg2
import os
import re
from WorthyPar.Query import queries_analysis

Primary = {"aka_name": "an",
           "aka_title": "at",
           "cast_info": "ci",
           "char_name": "chn",
           "comp_cast_type": "cct",
           "company_name": "cn",
           "company_type": "ct",
           "complete_cast": "cc",
           "info_type": "it",
           "keyword": "k",
           "kind_type": "kt",
           "link_type": "lt",
           "movie_companies": "mc",
           "movie_info_idx": "mi_idx",
           "movie_keyword": "mk",
           "movie_link": "ml",
           "name": "n",
           "role_type": "rt",
           "title": "t",
           "movie_info": "mi",
           "person_info": "pi"}


def extract_info(dbname, user, password):
    Tables = extract_db(dbname, user, password)
    Queries = extract_q(dbname, Tables)
    Locations = extract_loc(dbname, Tables, Queries, user, password)
    return generate_code(Tables, Queries, Locations), Tables


def get_table_size(dbname,table):
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database=dbname,
        user="css",
        password="postgres"
    )
    cur = conn.cursor()
    cur.execute(f"""SELECT (CAST(pg_total_relation_size('{table}') AS float) / (1024.0 * 1024.0))   """)
    return cur.fetchone()[0]


def get_queries_all(dbname):
    queries = dict()
    root_path = os.path.dirname(os.getcwd())
    relative_path = "data/" + dbname + "/queries/"
    folder_path = os.path.join(root_path, relative_path)
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r') as f:
            query_name, _ = os.path.splitext(file)
            lines = f.readlines()
            merged_lines = ''.join(lines).replace('\n', ' ')
            merged_lines = re.sub(r'\s+', ' ', merged_lines)
            queries[query_name] = merged_lines
    return queries


def get_queries(dbname):
    Queries = get_queries_name(dbname)
    queries = dict()
    root_path = os.path.dirname(os.getcwd())
    relative_path = "data/" + dbname + "/queries/"
    folder_path = os.path.join(root_path, relative_path)
    for file in Queries:
        file_path = os.path.join(folder_path, file+'.sql')
        with open(file_path, 'r') as f:
            query_name, _ = os.path.splitext(file)
            lines = f.readlines()
            merged_lines = ''.join(lines).replace('\n', ' ')
            merged_lines = re.sub(r'\s+', ' ', merged_lines)
            queries[query_name] = merged_lines
    return queries

def get_queries_name(dbname):
    queries = []
    root_path = os.path.dirname(os.getcwd())
    relative_path = "data/" + dbname + "/trainset/"
    folder_path = os.path.join(root_path, relative_path)
    files = os.listdir(folder_path)
    for file in files:
        query_name, _ = os.path.splitext(file)
        queries.append(query_name)
    return queries


def extract_db(dbname, user, password, host="localhost", port=5432):
    # Extract information for each table in the database
    # {"table_name":[table_size, tuple_num, column_num, column_names]}
    Tables = dict()
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=dbname,
        user=user,
        password=password
    )
    cur = conn.cursor()
    cur.execute("""
                    SELECT relname, (CAST(pg_relation_size(relid) AS float) / (1024.0 * 1024.0))   
                    FROM pg_stat_user_tables""")
    tables = cur.fetchall()
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table[0]}")
        tuple_num = cur.fetchone()[0]
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name='{table[0]}'")
        columns = cur.fetchall()
        column_names = [column[0] for column in columns]
        Tables[table[0]] = [table[1], tuple_num, len(column_names), column_names]
    print(Tables)
    cur.close()
    conn.close()
    return Tables


def extract_q(dbname, tables):
    queries = get_queries_all(dbname)
    # extract the table, select column, where column information involved in the query
    # Queries {"query_name": {"tables":[t1, t2, ...],
    #                         "select":[(t1, c1),...],
    #                         "where":[(t1, c1),...]}}
    Queries = queries_analysis(queries, tables)
    print(Queries)
    return Queries


def extract_loc(dbname, tables, queries, user, password, host="localhost", port=5432):
    Loctions = dict()
    queries_new = dict()
    root_path = os.path.dirname(os.getcwd())
    relative_path = "data/" + dbname + "/trainset/"
    folder_path = os.path.join(root_path, relative_path)
    print(folder_path)
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r') as f:
            query_name, _ = os.path.splitext(file)
            lines = f.readlines()
            merged_lines = ''.join(lines).replace('\n', ' ')
            merged_lines = re.sub(r'\s+', ' ', merged_lines).strip()
            queries_new[query_name] = merged_lines[:-1]
    for query_name, query_new in queries_new.items():
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=dbname,
            user=user,
            password=password
        )
        cur = conn.cursor()
        query_results = dict()
        for match_table in queries[query_name]["tables"]:
            query_result = dict()
            for column in tables[match_table][3]:
                match_table_new = f"(SELECT * FROM {match_table} ORDER BY {column}) AS match_new"
                cur.execute(f"""
                                        SELECT  distinct(CAST(CEIL(CAST(row_num AS FLOAT) * 10.0 / CAST({
                tables[match_table][1]} AS FLOAT)) AS INT))  
                                        FROM (  
                                                SELECT id ,row_number() OVER () AS row_num
                                                FROM {match_table_new}
                                        ) AS new_table
                                        WHERE id IN ( SELECT {Primary[match_table]}_id 
                                                                          FROM ({query_new}) AS query_result)""")
                query_result[column] = cur.fetchall()
            query_results[match_table] = result_convert(query_result)
        Loctions[query_name] = query_results
        print(query_name, query_results)
    return Loctions


def extract_parse_time(dbname, queries, user, password, host="localhost", port=5432):
    parseTimes = dict()
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=dbname,
        user=user,
        password=password
    )
    cur = conn.cursor()
    root_path = os.path.dirname(os.getcwd())
    relative_path = "data/" + dbname + "/queries/"
    folder_path = os.path.join(root_path, relative_path)
    for file in queries.keys():
        file_path = os.path.join(folder_path, file+'.sql')
        with open(file_path, 'r') as f:
            query_name, _ = os.path.splitext(file)
            lines = f.readlines()
            merged_lines = ''.join(lines).replace('\n', ' ')
            merged_lines = re.sub(r'\s+', ' ', merged_lines)
            cur.execute(f"EXPLAIN ANALYZE {merged_lines}")
            results = cur.fetchall()
            for row in results:
                if "Planning Time:" in row[0]:
                    parse_time = row[0].split(" ")[-2]
                    break
            parseTimes[query_name] = parse_time
    return parseTimes


def generate_code(tables, queries, locations):
    Info = dict()
    Queries = dict()
    db_list = []
    db_size = []
    db_query = dict()
    for name, information in tables.items():
        db_list.append([10, information[2]])
        db_size.append(information[0])
    Info["db_schema"] = db_list
    Info["db_size"] = db_size
    for name, _ in locations.items():
        information = queries[name]
        Queries[name] = information
        select = []
        for column in information['select']:
            table_name = column[0]
            column_name = column[1]
            if table_name != 'constant':
                select.append(
                    [list(tables.keys()).index(table_name), tables[table_name][3].index(column_name)])
        table = []
        for table_name in information['tables']:
            table.append(list(tables.keys()).index(table_name))
        where = []
        for column in information['where']:
            table_name = column[0]
            column_name = column[1]
            if table_name != 'constant':
                loc = []
                for loc_name, loc_where in locations[name][table_name].items():
                    loc.append([tables[table_name][3].index(loc_name), loc_where])
                where.append(
                    [list(tables.keys()).index(table_name), tables[table_name][3].index(column_name), loc])
        db_query[name] = [select, table, where]
    Info["db_query"] = db_query
    return Info, Queries


def result_convert(results):
    converted_results = dict()
    for name, result in results.items():
        converted_result = [item[0] for item in result]
        converted_results[name] = converted_result
    return converted_results


def print_info(info):
    for name, infomation in info.items():
        print(name, " : ", infomation)


