import pandas as pd
import numpy as np
DAY_SIZE = 24*60  #m
WEEK_SIZE = 7*24*60  #m
if __name__ == '__main__':
    # A1 = [1, 2, 3]
    # A2 = [4, 5, 6]
    # dic = {}
    # dic[1] = A1
    # dic[2] = A2
    # time = np.arange(0, WEEK_SIZE, 5).tolist()
    # temp = 5
    # # print(time)
    # df = pd.DataFrame(columns=['time'], data=[[0]])
    # for i in range(3-1):
    #
    #     df = df.append({'time': (i+1)*5}, ignore_index=True)
    #
    # print(df)
    # for temp_idx in dic.keys():
    #     temp_data = pd.Series(dic[temp_idx])
    #     df.insert(df.shape[1], str(temp_idx), temp_data, allow_duplicates=False)
    #
    # print(df)
    # dataa = df
    # # z_score = data.apply(lambda x : (x-x.mean())/x.std())
    # mean = dataa.mean()
    # std = dataa.std()
    # z_score = (dataa-mean)/std
    # print(z_score)
    #
    # dataa = z_score*std+mean
    #
    # print(dataa)
    #
    # dataa = dataa.drop(['time'], axis=1)
    # print(dataa)
    #
    # xx = [1,8,10,55,2,4]
    # xx = np.asarray(xx)
    # top_idx = xx.argsort()[-1:-6:-1]
    # print(top_idx)
    query_data = pd.read_csv('./dataset/query_data_test.csv')
    print(query_data)
    time = pd.timedelta_range(start='1 day', end='8 days', freq='5T')
    time = time[:-1]
    print(time)
    data = pd.date_range('2022-01-01', '2022-01-08', freq='5T')
    data = data[:-1]
    print(data)
    # query_data = query_data.set_index(time, drop=True)
    # print(query_data)
