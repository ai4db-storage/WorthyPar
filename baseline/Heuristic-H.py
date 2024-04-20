from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncode
import pandas as pd
class Heuristic_H():
    def __init__(self, info, index):
        self.info = info
        self.index = index
        self.min_support = 0.5
        self.attribute = self.extract_frequent_attribute()
        self.partition_num = 3

    def extract_frequent_attribute(self):
        dataset = []
        for query in self.info['db_query'].values():
            if self.index in query[1]:
                q_target = []
                for column in query[2]:
                    if column[0] == self.index:
                        q_target.append(column[1])
                q_rate = query[3]
                for i in range(int(q_rate*100)):
                    dataset.append(q_target)
        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset)
        dataframe = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_attributes = apriori(dataframe, min_support=self.min_support, use_colnames=True)
        attribute = frequent_attributes.loc[frequent_attributes['support'].idxmax()]
        return attribute

    def generate_partition(self, tables, path):
        table_name = list(tables.keys())[self.index]
        attribute = []
        for idx in self.attribute:
            attribute.append(tables[table_name][3][idx])
        partition_name = table_name + "_p1"
        print(f"CREATE TABLE {partition_name} AS ("
              f"SELECT * FROM {table_name} WHERE 1=0) temp"
              f"partition by HASH ({','.join(attribute)});")
        for i in range(self.partition_num):
            sub_partition_name = partition_name + str(i)
            print(f"create table {sub_partition_name} partition of {partition_name} for values with (modulus {self.partition_num}, remainder {i});")
        print(f"copy {partition_name} from '{path}/{table_name}.dat' with delimiter as '|' NULL '';")

