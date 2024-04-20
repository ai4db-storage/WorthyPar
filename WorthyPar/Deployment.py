# Deployment Module
def release_partition(partition_name):
    for pname in partition_name:
        print(f"DROP TABLE {pname}")


def generate_partition_worthypar(tables, query_name, index, order, row_block, col_block):
    table_name = list(tables.keys())[index]
    partition_name = table_name + "_" + query_name + "_p"
    start = -1
    row_last = 0
    where_condition_list = []
    for row in row_block:
        if start == -1:
            start = (row - 1) * int(tables[table_name][1] / 10)
            end = row * int(tables[table_name][1] / 10)
        elif row == row_last + 1:
            end = row * int(tables[table_name][1] / 10)
        else:
            where_condition_list.append("row_num BETWEEN " + str(start) + " AND " + str(end))
            start = (row - 1) * int(tables[table_name][1] / 10)
            end = row * int(tables[table_name][1] / 10)
        row_last = row

    where_condition_list.append("row_num BETWEEN " + str(start) + " AND " + str(end))
    where_condition = ' OR '.join(where_condition_list)
    order_key = tables[table_name][3][order]
    table_new = f"(SELECT * FROM {table_name} ORDER BY {order_key}) table_new"

    print(f"CREATE TABLE {partition_name} AS ("
          f"SELECT * FROM ("
          f"SELECT *, row_number() OVER() AS row_num "
          f"FROM {table_new}) temp WHERE {where_condition});")
    cols = set(range(tables[table_name][2])).difference(col_block)
    for col in cols:
        print(f"ALTER TABLE {partition_name} DROP COLUMN {tables[table_name][3][col]};")
    print(f"ALTER TABLE {partition_name} DROP COLUMN row_num;")
    print(f"VACUUM FULL {partition_name};")
    return partition_name
