## Balanced Data Parallel

Original data parallel fails to make the most of every gpu because the maximum batch size per gpu is given by GPU 0 while some space of GPU 0 is used to accumulate outputs, gradients, etc.

Thus, we just make batch sizes for other gpus bigger and keep the batch size for GPU 0 the same.

|          Name          	|    Url                 |
|:----------------------:	|:---------------------: |
|      Data Parallel     	| [🔗](https://github.com/sherlcok314159/dl-tools/blob/main/balanced_data_parallel/orig_dp.py) 	|
| Balanced Data Parallel 	|  [🔗](https://github.com/sherlcok314159/dl-tools/blob/main/balanced_data_parallel/main.py)  	|

