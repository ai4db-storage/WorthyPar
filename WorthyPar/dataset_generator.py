import argparse
from query_workload_generator import Query_Workload_Generator
import datetime

def parse_arg():
    args = argparse.ArgumentParser()
    # params for dataset
    # args.add_argument('--path', default='../dataset')
    # args.add_argument('--measurement', default='data')
    args.add_argument('--monitor_interval', default='5m')
    args.add_argument('--time_size', default='WEEK_SIZE')
    args.add_argument('--query_num', default=99)

    # args for workload generator
    args.add_argument('--workload_pattern', default='Cycle', choices=['Cycle', 'Spike', 'Evolution'])
    args.add_argument('--fix_pattern', default=True)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()

    query_generator = Query_Workload_Generator(args)
    query_generator.query_generate()
    # query_generator.query_stats()
    query_generator.save_queries(path='./dataset')
