# Evaluation Module
import random
import csv
from WorthyPar.Extraction import extract_info, extract_parse_time
from WorthyPar.Agents import generate_result
from WorthyPar.Deployment import release_partition
from WorthyPar.Evaluation import generate_query_plans, execution_query_plans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split as TTS


def generate_dataset(dbname, size=1000):
    dataset = []
    Info, Tables, Queries = extract_info(dbname)
    parseTimes = extract_parse_time(dbname, Queries)
    for i in range(size):
        # 生成workload
        workloadSize = random.randint(20, 1000)
        workload = []
        parseTime = 0
        selectNum = 0
        fromNum = 0
        whereNum = 0
        queryNum = dict()
        for i in range(workloadSize):
            index = random.randint(0, len(Info['db_query']) - 1)
            query_name = list(Info['db_query'].keys())[index]
            workload.append(query_name)
            parseTime += float(parseTimes[query_name])
            selectNum += len(Info['db_query'][query_name][0])
            fromNum += len(Info['db_query'][query_name][1])
            whereNum += len(Info['db_query'][query_name][2])
            if query_name in queryNum.keys():
                queryNum[query_name] += 1
            else:
                queryNum[query_name] = 1
        for query_name, query_code in Info['db_query'].items():
            Info["db_query"][query_name].append(0)
        for query_name, query_size in queryNum.items():
            rate = query_size * 1.0 / workloadSize
            Info["db_query"][query_name][-1] = rate
        # 生成分区方案
        # train(Info)
        Result, blockSize, blockNum, partitionNum = generate_result(Info)
        partition_name = []
        """
        for query_name, result in Result.items():
            for r in result:
                partition_name.append(generate_partition_worthypar(Tables, query_name, r[0], r[1], r[2], r[3]))
        """
        query_plans, scan_size = generate_query_plans(dbname, 'WorthyPar', workload)
        print('scan_size(MB):', scan_size)
        runtime = execution_query_plans(dbname, query_plans, 1)
        print('runtime(s):', runtime / 1000)
        release_partition(partition_name)
        dataset.append([workloadSize, len(queryNum), parseTime,
                        selectNum, fromNum, whereNum,
                        blockSize, blockNum, partitionNum, runtime])
    return dataset


def predict(path):
    data = pd.read_csv(path)
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    Xtrain, Xtest, Ytrain, Ytest = TTS(X, Y, test_size=0.1, random_state=420)

    reg_mod = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.08,
        subsample=0.75,
        colsample_bytree=1,
        max_depth=7,
        gamma=0,
    )

    eval_set = [(Xtrain, Ytrain), (Xtest, Ytest)]
    reg_mod.fit(Xtrain, Ytrain, eval_set=eval_set, eval_metric='rmse', verbose=False)

    scores = cross_val_score(reg_mod, Xtrain, Ytrain, cv=10)
    print("Mean cross-validation score: %.2f" % scores.mean())

    predictions = reg_mod.predict(Xtest)
    rmse = np.sqrt(mean_squared_error(Ytest, predictions))
    print("RMSE: %f" % (rmse))
    r2 = np.sqrt(r2_score(Ytest, predictions))
    print("R_Squared Score : %f" % (r2))

    # Loss
    sns.set_style("white")
    color = ["#344987","#C1E1EB"]
    palette = sns.color_palette(color)
    plt.plot(reg_mod.evals_result()['validation_0']['rmse'], label='train', color=palette[0], linewidth=2)
    plt.plot(reg_mod.evals_result()['validation_1']['rmse'], label='test', color=palette[1], linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('Loss.png', dpi=500)
    plt.show()

    # Fitting
    sns.set_style("white")
    palette = sns.color_palette(color)
    x_ax = range(len(Ytest))
    plt.plot(x_ax, Ytest, label="True Values", color=palette[0], linewidth=1)
    plt.plot(x_ax, predictions, label="Predicted Values", color=palette[1], linewidth=1)
    plt.xlabel("Sample Number")
    plt.ylabel("Scan Size")
    plt.legend()
    plt.savefig('True vs Predicted Values.png', dpi=500)
    plt.show()


def save_dataset(path, dataset):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for data in dataset:
            writer.writerow(data)


if __name__ == '__main__':
    dbname = 'imdb'
    path = 'output.csv'
    # dataset = generate_dataset(dbname)
    # save_dataset(path, dataset)
    predict(path)
