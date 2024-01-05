## Distributed Data Parallel

This folder just provides my example code for distributed data parallel (DDP).

|          Name          	|    Url                 |
|:----------------------:	|:---------------------: |
|      DDP-Launch     	| [🔗](https://github.com/sherlcok314159/dl-tools/blob/main/distributed_data_parallel/main_launch.py) 	|
| DDP-mp 	|  [🔗](https://github.com/sherlcok314159/dl-tools/blob/main/distributed_data_parallel/main_mp.py)  	|

How to use them is really easy. 

To launch a DDP:
```bash
cd distributed_data_parallel
python -m torch.distributed.launch --nproc_per_node=2 main_launch.py   
```

To spawn a DDP:
```bash
cd distributed_data_parallel
python main_mp.py
```


