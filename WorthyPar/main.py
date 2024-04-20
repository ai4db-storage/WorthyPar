from WorthyPar.Extraction import extract_info
from WorthyPar.Agents import train, generate_result
from WorthyPar.Deployment import generate_partition_worthypar
from WorthyPar.Evaluation import generate_workload, generate_query_plans, execution_query_plans
if __name__ == '__main__':
    # Please replace with your database username and password
    dbname = 'imdb'
    user = 'css'
    password = "postgres"
    Info, Queries, Tables= extract_info(dbname, user, password)
    # Set the query frequency
    num = float(len(Info['db_query']))
    for query_name, query_code in Info['db_query'].items():
        query_code.append(1 / num)
    # train(Info)
    Result, _, _, _ = generate_result(Info)
    for query_name, result in Result.items():
        for r in result:
            generate_partition_worthypar(Tables, query_name, r[0], r[1], r[2], r[3])
    for i in [20, 50, 100, 200, 500, 1000]:
        workload = generate_workload(dbname, i)
        print('workload', workload)
        query_plans, scan_size = generate_query_plans(dbname, 'WorthyPar', workload, Queries)
        runtime = execution_query_plans(dbname, query_plans, 3)
        print('scan_size(MB):', scan_size)
        print('runtime(s):', runtime / 1000)







