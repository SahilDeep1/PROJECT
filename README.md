# PROJECT

DS -222 FINAL PROJECT

Project Title : Collaborative Filtering for Recommender Systems

TO start in Local mode for any dataset MovieLens 100k, 1M ,10M ,20M

$python3 main_local.py --data_path=<\path of data file> --latent_factors<Add number> --batch_size=<Add Batch Size> --baseline=<add either 0(with matrix Factorization) or 1(without matrix Factorization)>

Example

 $python3 main_local.py --data_path=../data/ml-20m/ratings.csv ,latent_factors =10 --batch_size=50000 --baseline=0

To start using RBM implementation use:

$python3 r.py --data_path=<path of data file ratings >

To start in Distributed mode use the following commands:-

 Run the ps on Machine 1
 
$python3 async/sync/ssp.py --data_path=<Path of ratings file> --ps_hosts=<ip_ps_1>:<port_ps_1> --worker_hosts=<ip_worker_1:port_worker_1>,<ip_worker2:port_worker_2> --job_name=ps --task_index=0
 
 Run worker 1 on Machine 2
 
$python3 async/sync/ssp.py --data_path=<Path of ratings file> --ps_hosts=<ip_ps_1>:<port_ps_1> --worker_hosts=<ip_worker_1:port_worker_1>,<ip_worker2:port_worker_2> --job_name=worker --task_index=0

 Run worker 2 on Machine 3
 
$python3 async/sync/ssp.py --data_path=<Path of ratings file> --ps_hosts=<ip_ps_1>:<port_ps_1> --worker_hosts=<ip_worker_1:port_worker_1>,<ip_worker2:port_worker_2> --job_name=worker --task_index=1

For Data Analysis use Jupyter notebook open 1.ipynb or 2.ipynb
as 

$ Jupyter notebook 

