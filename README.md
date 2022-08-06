# TiCoSeRec
Source code for AAAI 2023 paper: [Uniform Sequence Better: Time Interval Aware Data Augmentation for Sequential Recommendation]()

In this paper, we explore the impact of time interval on SR models. The empirical study demonstrates that user preferences can be better learned on uniform sequence than non-uniform sequence in time dimension by SR models. Based on the previous works and analysis, we propose time interval aware data augmentation methods for the sequential recommendation, including a data augmentation restriction for user sequence and five time interval aware augmentation operators.

# Implementation
## Environment

Python >= 3.7

torch == 1.11.0+cu113 (We haven't tested the code on the lower version of torch)

numpy == 1.20.1

gensim = 4.2.0

tqdm == 4.59.0

pandas == 1.2.4

## Datasets

- Processed Sports dataset is included in `data` folder. 

  XXX_org_rank: users will maintain the original order.

  XXX_var_rank: users will be ranked by the variance of the interaction time interval.

- You can use the code in `data_process` folder to process your own dataset, and we explained its role at beginning of each code file.

## Train Model

- Delete in `_org` or `_var` the file name. 
  
  Example: If you want to use the `Sports` dataset ranked by variance, change the `Sports_item_var_rank.txt` into `Sports_item_rank.txt`, change the `Sports_time_var_rank.txt` into `Sports_time_rank.txt`.

- Change to `src` folder and Run the following command. (The program will read the data file according to [DATA_NAME]. [Model_idx] and [GPU_ID] can be specified according to your needs)
  
  ```
  python main.py --data_name=[DATA_NAME] --model_idx=[Model_idx] --gpu_id=[GPU_ID]
  ```

  ```
  Example:
  python main.py --data_name Sports --model_idx 1 --gpu_id 0
  ```

- The code will output the training log, the log of each test, and the `.pt` file of each test. You can change the test frequency in `src/main.py`.
- The meaning and usage of all other parameters have been clearly explained in `src/main.py`. You can change them as needed.

## Evaluate Model

- Change to `src` folder, Move the `.pt` file to the `src/output` folder. We give the weight file of the Spotrs dataset.

- Run the following command.
  ```
  python main.py --data_name=[DATA_NAME] --eval_path=[EVAL_PATH] --do_eval --gpu_id=[GPU_ID]
  ```

  ```
  Example:
  python main.py --data_name Sports --eval_path=./output/Sports.pt --do_eval --gpu_id=0
  Results:
  'HIT@5': '0.0319', 'NDCG@5': '0.0214', 'HIT@10': '0.0498', 'NDCG@10': '0.0271', 'HIT@20': '0.0752', 'NDCG@20': '0.0335'
  ```

# Acknowledgement
 - Training pipeline is implemented based on [CoSeRec](https://github.com/YChen1993/CoSeRec).
 - SASRec model are implemented based on [RecBole](https://github.com/RUCAIBox/RecBole). 

Thanks them for providing efficient implementation.

# Reference

Please cite our paper if you use this code.

```
latex bib reference
```
