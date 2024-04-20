# WorthyPar
WorthyPar is a workload-aware data hybrid partitioning advisor with deep reinforcement learning.
# Structure
```
- WorthyPar(main repo)
  - main.py
  - Exraction.py(extract the information of tables and workloads) 
    Query.py(parse sql, extract the information of sql[select, from, where])
  - Agents.py(generate hybrid partitioning strategies)
    ENV.py(the environment of the agents)
    Performance.py(analyze the performance of partitioning strategy.)
  - Deployment.py(generate partitioned table)
  - Evaluation.py(get runtime and scan size)
- data(import.sql and queries)
- model(trained network model)
- baseline(algorithms for comparison)
```
# Setup
The repository is developed with python 3.7 and pytorch 1.13.1 and PostgreSQL 15.4
```
conda create -n partition python=3.7
pip install -r requirements.txt
```
# Dataset
(1) Please download the data according to the link given(tpc toolkits / JOB dataset)
    The queries is in /data/queries
- TPC-DS (SF=30) https://gitcode.com/gregrahn/tpch-kit/blob/master/README.md
- TPC-H (SF=30)  https://gitcode.com/gregrahn/tpcds-kit/blob/master/README.md
- IMDB           https://gitcode.com/gregrahn/join-order-benchmark/blob/master/README.md
(2) Install PostgreSQL 15.4, create databases tpcds, tpch, and imdb, and import data to the database
    Execute the.sql file in the data folder to create the table and import the data
    Replace the path of the copy instruction in the import.sql file with the location of the generated data      
# Execution
- `main.py`: to generate hybrid partitioning strategy
# Workload Forecasting
* run `python forecaster_models.py` Utilizing the template design pattern, complete the creation of the base sequence prediction model class.
* run `python forecaster.py` training time series model and forecasting.
* run `python pipeline.py` Complete data reading and model training and prediction.

