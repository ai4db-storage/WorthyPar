import sqlparse
import sqlglot
import sqlglot.expressions as exp
from pprint import pprint
import random
import re
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML


def queries_analysis(queries, tables):
    Queries = dict()
    for name, sql in queries.items():
        table_names, select_columns, where_columns = query_analysis(sql, tables)
        Queries[name] = {"tables": table_names, "select": select_columns, "where": where_columns}
    return Queries


def query_analysis(sql, actual_tables):
    global tables, cte_tables, alias_columns, alias_tables, temp_table_num, where_columns, query_sql
    tables = actual_tables
    cte_tables = dict()
    alias_tables = dict()
    alias_columns = dict()
    temp_table_num = 0
    where_columns = []
    if re.search(r'with (.*) select', sql, re.DOTALL | re.IGNORECASE):
        process_with(sql)
    table_names = extract_table_name(sql)
    select_columns = extract_select_columns(sql)
    where_columns.extend(extract_where_columns(sql))
    where_columns = list(set(where_columns))
    return table_names, select_columns, where_columns


def extract_table_name(sql):
    tables = []
    for table in sqlglot.parse_one(sql).find_all(exp.Table):
        if "AS" in table.sql():
            origin, alias = get_table_aliasName(table.sql())
            alias_tables.update({alias: origin})
        if table.name not in cte_tables.keys():
            tables.append(table.name)
    return list(set(tables))


def extract_select_columns(sql):
    select_columns = analysis_select(sql)
    find_all_alias(sql)
    result_columns = []
    for key, values in select_columns.items():
        result_columns.extend(values)
    return list(set(result_columns))


def extract_where_columns(sql):
    where_columns = []
    for where in sqlglot.parse_one(sql).find_all(exp.Where):
        where = where.sql()
        # print("[extract_where_columns] where sql: ", where)
        index = where.find("WHERE", 0) + len("WHERE")
        condition_list = find_all_and_conditions(where[index:])
        # print("[extract_where_columns] conditions_list: ", condition_list)
        for condition in condition_list:
            # print("[extract_where_columns] condition: ", condition)
            where_columns.extend(get_tokens(condition))

    for group in sqlglot.parse_one(sql).find_all(exp.Group):
        group = group.sql()
        group_list = re.findall(r'GROUP BY\s+(.+)', group)
        groups = find_all_columns(group_list[0])
        for group_col in groups:
            group_col = group_col.strip()
            where_columns.extend(extract_column(group_col))
    for order in sqlglot.parse_one(sql).find_all(exp.Order):
        order = order.sql()
        order_list = re.findall(r'ORDER BY\s+(.+)', order)
        orders = find_all_columns(order_list[0])
        for order_col in orders:
            order_col = order_col.strip()
            if " desc" in order_col:
                order_col = order_col[:order_col.find("desc", 0)].strip()
            if " asc" in order_col:
                order_col = order_col[:order_col.find("asc", 0)].strip()
            if "case" in order_col:
                order_col, _ = process_case(order_col)
            where_columns.extend(extract_column(order_col))
    for having in sqlglot.parse_one(sql).find_all(exp.Having):
        having = having.sql()
        index = having.find("HAVING", 0) + len("HAVING")
        where_columns.extend(get_tokens(having[index:]))
    result_columns = []
    for token in where_columns:
        origin_columns = find_alias_column(token)
        for origin_column in origin_columns:
            result_columns.extend(find_which_table(origin_column))
    return list(set(result_columns))


def extract_from_part(parsed):
    from_seen = False
    for item in parsed.tokens:
        if from_seen:
            if is_subselect(item):
                for x in extract_from_part(item):
                    yield x
            elif item.ttype is Keyword:
                from_seen = False
                continue
            else:
                yield item
        elif item.ttype is Keyword and item.value.upper() == 'FROM':
            from_seen = True


def extract_table_identifiers(token_stream):
    for item in token_stream:
        if isinstance(item, IdentifierList):
            for identifier in item.get_identifiers():
                yield identifier.get_name()
        elif isinstance(item, Identifier):
            yield item.get_name()
        elif item.ttype is Keyword:
            yield item.value


def extract_column(str):
    str = get_contain(str)
    result = []
    for column in find_all_columns(str):
        if "rank()" in column:
            result.extend(process_rank(column))
        elif " over " in column:
            result.extend(process_over(column))
        elif is_expression(column):
            result.extend(get_column(column))
        else:
            result.extend(get_aggregateName(column))
    return result


def process_with(sql):
    match = re.search(r'with (.+)', sql, re.IGNORECASE)
    temp_tables, _ = find_all_with(match.group(1))
    for i in range(len(temp_tables)):
        cte_name = temp_tables[i][0]
        condition = temp_tables[i][1]
        cte_columns = dict()
        select_columns = analysis_select(condition)
        for key, values in select_columns.items():
            if "." in key:
                key = key.split(".")[1]
            cte_columns.update({key: values})
        cte_tables.update({cte_name: cte_columns})


def process_case(sql):
    sql = sql.lower()
    where_list = []
    replaced_text = []
    if re.search(r'case when (.+) then (.+) else (.+) end', sql):
        case_list = re.findall(r'case when (.+) then (.+) else (.+) end', sql)
        for case_item in case_list:
            # print("[process_case] case_item: ", case_item)
            condition_list = find_all_and_conditions(case_item[0])
            for condition in condition_list:
                where_list.extend(get_tokens(condition))
            if "select " in case_item[1]:
                result1 = ",".join(find_all_selects(case_item[1]))
            else:
                result1 = case_item[1]
            if "select " in case_item[2]:
                result2 = ",".join(find_all_selects(case_item[2]))
            else:
                result2 = case_item[2]
            replaced_text.append("(" + result1 + "," + result2 + ")")
        new_sql = re.sub(r'case when (.+) then (.+) else (.+) end', lambda _: replaced_text.pop(0), sql)
    elif re.search(r'case when (.+) then (.+) end', sql):
        case_list = re.findall(r'case when (.+) then (.+) end', sql)
        for case_item in case_list:
            # print("[process_case] case_item: ", case_item)
            contain = get_contain(case_item[0])
            where_list.extend(get_tokens(contain))
            if "select " in case_item[1]:
                result1 = ",".join(find_all_selects(case_item[1]))
            else:
                result1 = case_item[1]
            replaced_text.append("(" + result1 + ")")
        new_sql = re.sub(r'case when (.+) then (.+) end', lambda _: replaced_text.pop(0), sql)
    elif re.search(r'case (.+) when (.+) then (.+) else (.+) end', sql):
        case_list = re.findall(r'case (.+) when (.+) then (.+) else (.+) end', sql)
        for case_item in case_list:
            # print("[process_case] case_item: ", case_item)
            where_list.append(case_item[0])
            if "select " in case_item[2]:
                result1 = ",".join(find_all_selects(case_item[2]))
            else:
                result1 = case_item[2]
            if "select " in case_item[3]:
                result2 = ",".join(find_all_selects(case_item[3]))
            else:
                result2 = case_item[3]
            replaced_text.append("(" + result1 + "," + result2 + ")")
        new_sql = re.sub(r'case (.+) when (.+) then (.+) else (.+) end', lambda _: replaced_text.pop(0), sql)
    elif re.search(r'case (.+) when (.+) then (.+) end', sql):
        case_list = re.findall(r'case (.+) when (.+) then (.+) end', sql)
        for case_item in case_list:
            # print("[process_case] case_item: ", case_item)
            where_list.append(case_item[0])
            if "select " in case_item[2]:
                result1 = ",".join(find_all_selects(case_item[2]))
            else:
                result1 = case_item[2]
            replaced_text.append("(" + result1 + ")")
        new_sql = re.sub(r'case (.+) when (.+) then (.+) end', lambda _: replaced_text.pop(0), sql)
    return new_sql, list(set(where_list))


def process_cast(str):
    str = str.lower()
    cast_start = str.find("cast", 0)
    _, index = match_bracket(str[cast_start:])
    cast_end = index + cast_start
    while cast_start != -1:
        cast = str[cast_start:cast_end]
        match = re.search(r'cast\((.+) as (.+)\)$', cast)
        str = str[:cast_start] + match.group(1) + str[cast_end:]
        cast_start = str.find("cast", 0)
        if cast_start != -1:
            _, index = match_bracket(str[cast_start:])
            cast_end = index + cast_start
    return str


def process_extract(str):
    str = str.lower()
    cast_start = str.find("extract", 0)
    _, index = match_bracket(str[cast_start:])
    cast_end = index + cast_start
    while cast_start != -1:
        cast = str[cast_start:cast_end]
        match = re.search(r'extract\((.+) from (.+)\)$', cast)
        str = str[:cast_start] + match.group(2) + str[cast_end:]
        cast_start = str.find("extract", 0)
        if cast_start != -1:
            _, index = match_bracket(str[cast_start:])
            cast_end = index + cast_start
    return str


def process_rank(str):
    str = str.lower()
    columns = []
    if re.search(r'rank\(\) over \(partition by (.+) order by (.+)\)$', str):
        match = re.search(r'rank\(\) over \(partition by (.+) order by (.+)\)$', str)
        columns.extend(extract_column(match.group(1)))
        columns.extend(extract_column(match.group(2)))
    elif re.search(r'rank\(\) over \(order by (.+)\)$', str):
        match = re.search(r'rank\(\) over \(order by (.+)\)$', str)
        columns.extend(extract_column(match.group(1)))
    elif re.search(r'rank\(\) over \(partition by (.+)\)$', str):
        match = re.search(r'rank\(\) over \(partition by (.+)\)$', str)
        columns.extend(extract_column(match.group(1)))
    return columns


def process_over(str):
    str = str.lower()
    columns = []
    if re.search(r'(.+) over \(partition by (.+) order by (.+) rows between unbounded preceding and current row\)$',
                 str):
        match = re.search(
            r'(.+) over \(partition by (.+) order by (.+) rows between unbounded preceding and current row\)$', str)
        columns.extend(extract_column(match.group(1)))
        columns.extend(find_all_columns(match.group(2)))
        columns.extend(find_all_columns(match.group(3)))
    elif re.search(r'(.+) over \(partition by (.+) order by (.+)\)$', str):
        match = re.search(r'(.+) over \(partition by (.+) order by (.+)\)$', str)
        columns.extend(extract_column(match.group(1)))
        columns.extend(find_all_columns(match.group(2)))
        columns.extend(find_all_columns(match.group(3)))
    elif re.search(r'(.+) over \(order by (.+)\)$', str):
        match = re.search(r'(.+) over \(order by (.+)\)$', str)
        columns.extend(extract_column(match.group(1)))
        columns.extend(extract_column(match.group(2)))
    elif re.search(r'(.+) over \(partition by (.+)\)$', str):
        match = re.search(r'(.+) over \(partition by (.+)\)$', str)
        columns.extend(extract_column(match.group(1)))
        columns.extend(extract_column(match.group(2)))
    return columns


def process_expression(str):
    str = get_contain(str.lower())
    count = 0
    index = 0
    if "cast" in str:
        expression = process_cast(str)
        # print("[process_expression] expression: ", expression)
        return process_expression(expression)
    if " as " in str:
        split_string = re.split(" as ", str)
        str = split_string[0].strip()
    if "'" in str:
        count, _, index = match_quotation(str)
    if (re.search(r'[-+*/]', str) and count != 1) or (count == 1 and index != len(str)):
        return get_column(str)
    if is_column(str):
        return str
    else:
        return []


def analysis_select(sql):
    global query_sql
    select_columns = dict()
    rename = dict()
    depth = 1
    select_node = sqlglot.parse_one(sql)
    if select_node.this:
        if select_node.key == 'union' or select_node.key == 'intersect':
            union_table = dict()
            while select_node.key == 'union' or select_node.key == 'intersect':
                for key, value in analysis_select(select_node.this.sql()).items():
                    if "." in key:
                        key = key.split(".")[1]
                    if key in union_table.keys():
                        union_table[key] += value
                    else:
                        union_table.update({key: value})
                select_node = select_node.right
            for key, value in analysis_select(select_node.sql()).items():
                if "." in key:
                    key = key.split(".")[1]
                if key in union_table.keys():
                    union_table[key] += value
                else:
                    union_table.update({key: value})
            return union_table
    from_table = dict()
    for from_node in select_node.find_all(exp.From):
        if from_node.depth != depth:
            continue
        if from_node.this.key == 'subquery':
            sub_query = from_node.this
            select_table = analysis_select(sub_query.this.sql())
            subquery_table = dict()
            for key, values in select_table.items():
                if "." in key:
                    key = key.split(".")[1]
                subquery_table.update({key: values})
            from_table.update(subquery_table)
            if sub_query.alias:
                cte_name = sub_query.alias
                cte_tables.update({cte_name: subquery_table})
        else:
            table = from_node.this.name
            if from_node.this.alias:
                alias_tables.update({from_node.this.alias: table})
            for column in find_table_columns(table):
                if column in from_table.keys():
                    rename[column].append(table)
                else:
                    from_table.update({column: find_which_table(column)})
                    rename.update({column: [table]})
    for from_node in select_node.find_all(exp.Join):
        if from_node.depth != depth:
            continue
        if 'on' in from_node.args:
            join_columns = []
            result_columns = []
            conditions = from_node.args['on'].sql()
            condition_list = find_all_and_conditions(get_contain(conditions))
            # print(condition_list)
            for condition in condition_list:
                join_columns.extend(get_tokens(condition))
            for token in join_columns:
                origin_columns = find_alias_column(token)
                for origin_column in origin_columns:
                    result_columns.extend(find_which_table(origin_column))
            where_columns.extend(result_columns)
        if from_node.this.key == 'subquery':
            sub_query = from_node.this
            sub_query_table = analysis_select(sub_query.this.sql())
            from_table.update(sub_query_table)
            if sub_query.alias:
                cte_name = sub_query.alias
                cte_tables.update({cte_name: sub_query_table})
        else:
            table = from_node.this.name
            if from_node.this.alias:
                alias_tables.update({from_node.this.alias: table})
            for column in find_table_columns(table):
                if column in from_table.keys():
                    rename[column].append(table)
                else:
                    from_table.update({column: find_which_table(column)})
                    rename.update({column: [table]})
    for column, tables in rename.items():
        if len(tables) > 1:
            from_table.pop(column)
            for table in tables:
                column_name = table + '.' + column
                from_table.update({column_name: find_which_table(column)})
    new_select_columns = []
    for select_column in select_node.selects:
        select_column = select_column.sql()
        if select_column.strip() == '*':
            select_columns = from_table
        else:
            if ("CAST" in select_column or "cast" in select_column) and ("CAST_" not in select_column and "cast_" not in select_column):
                select_column = process_cast(select_column)
            if "CASE" in select_column or "case" in select_column and ("CASE_" not in select_column and "case_" not in select_column):
                select_column, where_list = process_case(select_column)
                for token in where_list:
                    origin_columns = find_alias_column(token)
                    for origin_column in origin_columns:
                        where_columns.extend(find_which_table(origin_column))
            if "EXTRACT" in select_column or "extract" in select_column:
                select_column = process_extract(select_column)
            tokens = []
            columns = []
            column_name = select_column
            if "AS" in select_column or " as " in select_column:
                origin, alias = get_column_aliasName(select_column)
                if alias in alias_columns.keys():
                    alias_columns[alias].extend(origin)
                else:
                    alias_columns.update({alias: origin})
                column_name = alias
                tokens.extend(origin)
            else:
                tokens.extend(extract_column(select_column))
            for token in tokens:
                if token in from_table.keys():
                    columns.extend(from_table[token])
                else:
                    columns.extend(find_which_table(token))
            select_columns.update({column_name: list(set(columns))})
            new_select_columns.extend(list(set(columns)))
    return select_columns


def find_all_with(str):
    temp_tables = []
    while True:
        name, con, flag, str = match_with(str)
        temp_tables.append([name, con])
        if not flag:
            break
    return temp_tables, str


def find_all_selects(sql):
    sql = sql.lower()
    columns_list = []
    index = 0
    while True:
        select_index = sql.find("select", index)
        if select_index == -1:
            break
        from_index = sql.find("from", index)
        columns = sql[select_index + 6:from_index].strip()
        columns_list.append(columns)
        index = from_index + 4
    return columns_list


def find_all_columns(str):
    str = str.lower()
    column_list = []
    column = []
    i = 0
    while i < len(str):
        if str[i] == '(':
            _, index = match_bracket(str[i:])
            column.append(str[i:i + index])
            i = i + index
            if i == len(str):
                break
        if str[i:i + 4] == 'cast':
            _, index = match_bracket(str[i:])
            column.append(str[i:i + index])
            i = i + index
        if str[i] == ',':
            column_list.append(''.join(column).strip())
            # print("[find_all_columns] column: ", ''.join(column).strip())
            column = []
        else:
            column.append(str[i])
        i = i + 1
    column_list.append(''.join(column).strip())
    return column_list


def find_all_and_conditions(str):
    str = str.lower()
    conditions_group = []
    start_index = 0
    while start_index < len(str):
        index = str.find(" and ", start_index) + 1
        if index == 0:
            conditions_group.append(str[start_index:].strip())
            break
        else:
            condition = str[start_index:index]
            while not is_match(condition):
                index = str.find(" and ", index + 3) + 1
                if index == 0:
                    index = len(str)
                    break
                condition = str[start_index:index]
            conditions_group.append(str[start_index:index].strip())
            start_index = index + 3
    return conditions_group


def find_all_or_conditions(str):
    str = str.lower()
    conditions_group = []
    start_index = 0
    while start_index < len(str):
        index = str.find(" or ", start_index) + 1
        if index == 0:
            conditions_group.append(str[start_index:].strip())
            break
        else:
            condition = str[start_index:index]
            if not is_match(condition):
                while not is_match(condition):
                    index = str.find(" or ", index + 2) + 1
                    if index == 0:
                        conditions_group.append(str[start_index:].strip())
                        break
                    condition = str[start_index:index]
                if index == 0:
                    break
                conditions_group.append(str[start_index:index].strip())
            else:
                conditions_group.append(condition.strip())
            start_index = index + 2
    return conditions_group


def find_table_columns(table_name):
    if table_name in tables.keys():
        return tables[table_name][3]
    elif table_name in cte_tables.keys():
        return list(cte_tables[table_name].keys())
    else:
        print("[find_table_columns] ERROR: NOT FOUND", table_name)
        return []


def find_which_table(column):
    if not is_column(column):
        return [("constant", column)]
    if "." in column:
        table = column.split(".")[0].strip()
        column = column.split(".")[1].strip()
        if table in alias_tables.keys():
            table = alias_tables[table]
        if table in cte_tables.keys():
            return cte_tables[table][column]
        elif table in tables.keys():
            return [(table, column)]
        else:
            print("ERROR: UNDEFINED", column)

    for table_name in tables.keys():
        if column in tables[table_name][3]:
            return [(table_name, column)]

    for table_name in cte_tables.keys():
        if column in cte_tables[table_name].keys():
            return cte_tables[table_name][column]

    print("ERROR: 出现未定义的列名", column)


def find_alias_column(column):
    origin_columns = []
    flag = False
    if column in alias_columns.keys():
        for origin_column in alias_columns[column]:
            if origin_column != column:
                origin_columns.extend(find_alias_column(origin_column))
            else:
                flag = True
        if (origin_columns == []) and (alias_columns[column] != []) and flag:
            origin_columns.append(column)
    else:
        origin_columns.append(column)
    return origin_columns


def find_operator(str, start_index):
    index_list = []
    operators = ['+', '-', '*', '/']
    for operator in operators:
        index = str.find(operator, start_index)
        if index != -1:
            index_list.append(index)
    if not index_list:
        return -1
    else:
        return min(index_list)


def find_aggregate(str):
    result = []
    start_index = 0
    index = str.find(",", start_index)
    if index != -1:
        while not (is_match(str[:index].strip()) and is_match(str[index + 1:].strip())):
            start_index = start_index + index
            index = str.find(",", start_index)
            if index == -1:
                if is_expression(str):
                    return get_column(str)
                else:
                    return [str]
        if is_expression(str[:index].strip()):
            result.extend(get_column(str[:index].strip()))
        else:
            result.append(str[:index].strip())
        result.extend(find_all_aggregate(str[index + 1:].strip()))
        # print("[get_column] result: ", result)
        return result
    else:
        if is_expression(str):
            return get_column(str)
        else:
            return [str]


def find_all_aggregate(str):
    result = []
    for aggregate in find_aggregate(get_contain(str)):
        if aggregate.count("(") > 1:
            contain, _ = match_bracket(aggregate)
            result.extend(find_all_aggregate(contain))
        else:
            result.append(aggregate)
    return result


def find_all_alias(sql):
    for subquery in sqlglot.parse_one(sql).find_all(exp.Subquery):
        if subquery.alias and subquery.alias not in cte_tables.keys():
            #  print("[find_all_alias] subquery: ", subquery.alias)
            select_table = analysis_select(subquery.this.sql())
            subquery_table = dict()
            for key, values in select_table.items():
                if "." in key:
                    key = key.split(".")[1]
                subquery_table.update({key: values})
            cte_name = subquery.alias
            cte_tables.update({cte_name: subquery_table})


def is_expression(str):
    start_index = 0
    str = get_contain(str)
    index = find_operator(str, start_index)
    if index != -1:
        while not (is_match(str[:index].strip()) and is_match(str[index + 1:].strip())):
            start_index = index + 1
            index = find_operator(str, start_index)
            if index == -1:
                return False
        if ("," in str[:index].strip() and "(" not in str[:index].strip()) or ("," in str[index + 1:].strip() and "(" not in str[index + 1:].strip()):
            return False
        return True
    return False


def is_column(str):
    if re.match(r'^-?\d+(\.\d+)?([eE]-?\d+)?$', str):
        return False
    if "'" in str:
        return False
    if str == "null":
        return False
    if str == "distinct":
        return False
    if str == "asc":
        return False
    if str == "desc":
        return False
    return True


def is_digit(str):
    if re.match(r'^-?\d+(\.\d+)?([eE]-?\d+)?$', str):
        return True
    return False


def is_str(str):
    str = get_contain(str).strip()
    if str[0] == "'" and str[-1] == "'":
        return True
    return False


def is_subselect(parsed):
    if not parsed.is_group:
        return False
    for item in parsed.tokens:
        if item.ttype is DML and item.value.upper() == 'SELECT':
            return True
    return False


def is_match(str):
    flag = False
    if "between" in str:
        start_index = str.find("between", 0) + len("between")
        end_index = str.find("and", start_index)
        while end_index != -1:
            condition1 = str[start_index:end_index]
            if is_match(condition1):
                flag = True
                break
            end_index = str.find("and", end_index + 3)
        if not flag:
            return False
    if str.count('(') != str.count(')'):
        return False
    if str.count("'") % 2 != 0:
        return False
    return True


def match_with(str):
    name = str[:str.find(' ',0)]
    con, i = match_bracket(str)
    if str[i+1:i+7] == 'select':
        flag = False
    else:
        flag = True
        str = str[str.find(",", i)+1:].lstrip()
    return ''.join(name), ''.join(con)[:-1], flag, str


def match_bracket(str):
    i = str.find('(', 0)
    if i == -1:
        return str, len(str)
    contain = []
    num = 1
    while num > 0:
        i += 1
        if str[i] == '(':
            num += 1
        if str[i] == ')':
            num -= 1
        contain.append(str[i])
    return ''.join(contain)[:-1], i + 1


def match_quotation(str):
    num = 0
    result = []
    start_index = str.find("'", 0)
    end_index = 0
    while start_index != -1:
        end_index = str.find("'", start_index + 1)
        num += 1
        result.append(str[start_index:end_index + 1])
        start_index = str.find("'", end_index + 1)
    return num, result, end_index + 1


def get_tokens(where):
    where = get_contain(where)
    sql_tokens = []
    where_list = find_all_or_conditions(where)
    for where_item in where_list:
        where_item = get_contain(where_item)
        if " and " in where_item and "select" not in where_item and where_item.count(" and ") != where_item.count(
                " between "):
            for condition in find_all_and_conditions(where_item):
                sql_tokens.extend(get_tokens(condition))
        else:
            if " in " in where_item and is_match(where_item[:where_item.find(" in ", 0)]):
                index = where_item.find(" in ", 0)
                sql_tokens.extend(get_column(where_item[:index]))
            elif "exists" in where_item:
                continue
            elif re.search(r'^(.+) between (.+) and (.+)$', where_item) and "select" not in where_item:
                match = re.search(r'^(.+) between (.+) and (.+)$', where_item)
                if "case" in match.group(1):
                    _, where_list_between = process_case(match.group(1))
                    sql_tokens.extend(where_list_between)
                else:
                    sql_tokens.append(match.group(1))
                sql_tokens.extend(process_expression(match.group(2)))
                sql_tokens.extend(process_expression(match.group(3)))
            elif re.search(
                    r'^(.+) between \(select (.+) from (.+) where (.+)\) and \(select (.+) from (.+) where (.+)\)$',
                    where_item):
                match = re.search(
                    r'^(.+) between \(select (.+) from (.+) where (.+)\) and \(select (.+) from (.+) where (.+)\)$',
                    where_item)
                if "case" in match.group(1):
                    _, where_list_between = process_case(match.group(1))
                    sql_tokens.extend(where_list_between)
                else:
                    sql_tokens.append(match.group(1))

                for condition in find_all_and_conditions((match.group(4))):
                    sql_tokens.extend(get_tokens(condition))

                for condition in find_all_and_conditions((match.group(7))):
                    sql_tokens.extend(get_tokens(condition))

            elif re.search(r'^not (.+) is null$', where_item):
                match = re.search(r'^not (.+) is null$', where_item)
                sql_tokens.extend(get_column(match.group(1)))
            elif re.search(r'^(.+) is null$', where_item):
                match = re.search(r'^(.+) is null$', where_item)
                sql_tokens.extend(get_column(match.group(1)))
            elif re.search(r'^(.+) like (.+)$', where_item):
                match = re.search(r'^(.+) like (.+)$', where_item)
                sql_tokens.extend(get_column(match.group(1)))
            else:
                object1, object2 = conditions_split(where_item)
                if "select" in object1 and "where" in object1 and "group" in object1:
                    match = re.search(r'\(select (.+) from (.+) where (.+) group by (.+)\)', object1)
                    for condition in find_all_and_conditions((match.group(3))):
                        sql_tokens.extend(get_tokens(condition))
                    sql_tokens.extend(get_column(match.group(4)))
                elif "select" in object1 and "where" in object1:
                    match = re.search(r'\(select (.+) from (.+) where (.+)\)', object1)
                    for condition in find_all_and_conditions((match.group(3))):
                        sql_tokens.extend(get_tokens(condition))
                elif "select" in object1:
                    continue
                else:
                    if "cast" in object1 and "'cast'" not in object1:
                        object1 = process_cast(object1)
                    if "case" in object1 and "'case'" not in object1:
                        where_column1, where_list1 = process_case(object1)
                        sql_tokens.extend(get_column(where_column1))
                        sql_tokens.extend(where_list1)
                    else:
                        sql_tokens.extend(get_column(object1))

                if "select" in object2 and "where" in object2 and "group" in object2:
                    match = re.search(r'\(select (.+) from (.+) where (.+) group by (.+)\)', object2)
                    for condition in find_all_and_conditions((match.group(3))):
                        sql_tokens.extend(get_tokens(condition))
                    sql_tokens.extend(get_column(match.group(4)))
                elif "select" in object2 and "where" in object2:
                    match = re.search(r'\(select (.+) from (.+) where (.+)\)', object2)
                    for condition in find_all_and_conditions((match.group(3))):
                        sql_tokens.extend(get_tokens(condition))
                elif "select" in object2:
                    continue
                else:
                    if "cast" in object2 and "'cast'" not in object2:
                        object2 = process_cast(object2)
                    if "case" in object2 and "'case'" not in object2:
                        where_column2, where_list2 = process_case(object2)
                        sql_tokens.extend(get_column(where_column2))
                        sql_tokens.extend(where_list2)
                    else:
                        sql_tokens.extend(get_column(object2))
    return sql_tokens


def get_column(str):
    result = []
    if "not " in str:
        str = str.replace('not ', ' ').strip()
    if is_str(str.strip()) or is_digit(str.strip()):
        return result
    if is_expression(str):
        str = get_contain(str)
    start_index = 0
    index = find_operator(str, start_index)
    if index != -1:
        while not (is_match(str[:index].strip()) and is_match(str[index + 1:].strip())):
            start_index = index + 1
            index = find_operator(str, start_index)
            if index == -1:
                return get_aggregateName(str)
        if not is_str(str[:index].strip()):
            result.extend(get_column(str[:index].strip()))
        if not is_str(str[index + 1:].strip()):
            result.extend(get_column(str[index + 1:].strip()))
        return result
    if '(' in str:
        result.extend(get_aggregateName(str))
    else:
        if is_column(str.strip()):
            result.append(str.strip())
    return result


def get_aggregateName(str):
    result = []
    str = str.lower()
    aggregate_list = find_all_aggregate(str)
    for aggregate in aggregate_list:
        contain, _ = match_bracket(aggregate)
        for column in re.findall(r"[.'\w]+", contain):
            if is_column(column):
                result.append(column)
    return list(set(result))


def get_column_aliasName(str):
    str = str.lower()
    if " as " in str:
        split_string = re.split(" as ", str)
    elif re.search(r'(.+) over \((.+)\) (.+)', str):
        match = re.search(r'(.+) over \((.+)\) (.+)', str)
        result1 = extract_column(match.group(1))
        if re.search(r'partition by (.+) order by (.+) rows between unbounded preceding and current row',
                     match.group(2)):
            match1 = re.search(r'partition by (.+) order by (.+) rows between unbounded preceding and current row',
                               match.group(2))
            result1.extend(find_all_columns(match1.group(1)))
            result1.extend(find_all_columns(match1.group(2)))
        elif re.search(r'partition by (.+) order by (.+)', match.group(2)):
            match1 = re.search(r'partition by (.+) order by (.+)', match.group(2))
            result1.extend(find_all_columns(match1.group(1)))
            result1.extend(find_all_columns(match1.group(2)))
        elif re.search(r'order by (.+)', match.group(2)):
            match1 = re.search(r'order by (.+)', match.group(2))
            result1.extend(find_all_columns(match1.group(1)))
        else:
            match1 = re.search(r'partition by (.+)', match.group(2))
            result1.extend(find_all_columns(match1.group(1)))
        result2 = match.group(3).strip()
        return result1, result2
    else:
        split_string = re.split(" ", str)
    origin = split_string[0].strip()
    result1 = extract_column(origin)
    result2 = split_string[1].strip()
    return result1, result2


def get_table_aliasName(str):
    str = str.lower()
    if "as" in str:
        split_string = re.split("as", str)
    else:
        split_string = re.split(" ", str)
    result1 = split_string[0].strip()
    result2 = split_string[1].strip()

    return result1, result2


def get_contain(str):
    if str[0] == '(' and str[-1] == ')':
        _, index = match_bracket(str)
        if index == len(str):
            return str[1:-1]
    return str


def conditions_split(str):
    index = 0
    con = []
    while str[index] != '>' and str[index] != '<' and str[index] != '=':
        if str[index] == '(':
            num = 1
            while num > 0:
                con.append(str[index])
                index += 1
                if str[index] == '(':
                    num += 1
                if str[index] == ')':
                    num -= 1
            con.append(str[index])
            index += 1
        if str[index:index + 4] == 'case':
            while str[index:index + 4] != ' end':
                con.append(str[index])
                index += 1
            con.extend(str[index:index + 4])
            index = index + 4
        con.append(str[index])
        index += 1
    if str[index + 1] == '>' or str[index + 1] == '<' or str[index + 1] == '=':
        index += 1
    return ''.join(con).strip(), str[index + 1:].strip()

