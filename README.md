# TiCoSeRec
Official source code for AAAI 2023 paper: [Uniform Sequence Better: Time Interval Aware Data Augmentation for Sequential Recommendation](https://arxiv.org/abs/2212.08262)

In this paper, we explored the impact of time interval on sequential recommendations. Our basic idea is that uniform sequences are more valuable for next-item prediction. This assumption was validated by an empirical study. Then, we proposed five data operators to augment item sequences in the light of time intervals. Our experiments on four public datasets have verified the effectiveness of our proposed operators for data augmentation. To the authors' best knowledge, this is the first work to study the distribution of time interval for the sequential recommendation. For future work, we intend to further consider the factor of item category for data augmentation, and how time interval and item category can be leveraged together for better performance. 
# Implementation
## Environment

Python >= 3.7

torch == 1.11.0+cu113 (We haven't tested the code on the lower version of torch)

numpy == 1.20.1

gensim = 4.2.0

tqdm == 4.59.0

pandas == 1.2.4

## Datasets

- Processed Beauty, Sports and Home and two different version of Yelp datasets are included in `data` folder. 

  XXX_org_rank: users will maintain the original order.

  XXX_var_rank: users will be ranked by the variance of the interaction time interval.

- You can use the code in `data_process` folder to process your own dataset, and we explained its role at beginning of each code file.

- For Yelp dataset, we give two processed datasets from two different versions of Yelp. Yelp-A is processed from Yelp2020, which has 316,354 interactions. Yelp-B is processed from Yelp2022, which has 207,045 interactions.
## Train Model

- Delete `_rank_org` or `_rank_var` in the file name. 
  
  Example: If you want to use the Sports dataset ranked by variance, change the `Sports_item_var_rank.txt` into `Sports_item.txt`, change the `Sports_time_var_rank.txt` into `Sports_time.txt`.

- Change to `src` folder and Run the following command. (The program will read the data file according to [DATA_NAME]. [Model_idx] and [GPU_ID] can be specified according to your needs)
  
  ```
  python main.py --data_name=[DATA_NAME] --model_idx=[Model_idx] --gpu_id=[GPU_ID]
  ```

  ```
  Example:
  python main.py --data_name=Beauty --model_idx=1 --augmentation_warm_up_epochs=350 --mask_mode=maximum --substitute_rate=0.2 --crop_rate=0.4 --mask_rate=0.7 --reorder_rate=0.5 --gpu_id=0
  python main.py --data_name=Sports --model_idx=1 --substitute_rate=0.2 --weight_decay=1e-5 --patience=100 --gpu_id=0
  python main.py --data_name=Home --model_idx=1 --mask_mode=random --reorder_rate=0.4 --mask_rate=0.6 --patience=50 --weight_decay=1e-7 --gpu_id=0 
  ```

- The code will output the training log, the log of each test, and the `.pt` file of each test. You can change the test frequency in `src/main.py`.
- The meaning and usage of all other parameters have been clearly explained in `src/main.py`. You can change them as needed.

## Hyper-parameter Fine-Tuning
If you use your own dataset, we give some suggestions and ranges for fine-tuning of Hyper-parameters.
- augment_threshold: it needs to be adjusted according to the dataset. 
- augment_type_for_short: generally, `SIM` is better. You can try other operator combinations.
- ratio/rate for data augmentation operators: range `[0.1,0.9]` step by `0.1` or `0.2`.
- var_rank_not_aug_ratio: range `[0.1,0.5]` step by `0.1` or `0.05`.
- attn_dropout_prob and hidden_dropout_prob : range `[0.2,0.5]` step by `0.1`.
- weight_decay : range `[1e-4,1e-8]`.

## Evaluate Model

- Change to `src` folder, Move the `.pt` file to the `src/output` folder. We give the weight file of the Beauty, Sports and Home dataset.

- Run the following command.
  ```
  python main.py --data_name=[DATA_NAME] --eval_path=[EVAL_PATH] --do_eval --gpu_id=[GPU_ID]
  ```

  ```
  Example:
  python main.py --data_name=Beauty --eval_path=./output/Beauty.pt --do_eval --gpu_id=0
  python main.py --data_name=Sports --eval_path=./output/Sports.pt --do_eval --gpu_id=0
  python main.py --data_name=Home --eval_path=./output/Home.pt --do_eval --gpu_id=0
  Beauty Results:
  {'stage': 'test', 'epoch': 0, 'HIT@5': '0.0504', 'NDCG@5': '0.0343', 'HIT@10': '0.0740', 'NDCG@10': '0.0418', 'HIT@20': '0.1068', 'NDCG@20': '0.0501'}
  Sports Results:
  {'stage': 'test', 'epoch': 0, 'HIT@5': '0.0334', 'NDCG@5': '0.0227', 'HIT@10': '0.0514', 'NDCG@10': '0.0284', 'HIT@20': '0.0768', 'NDCG@20': '0.0348'}
  Home Results:
  {'stage': 'test', 'epoch': 0, 'HIT@5': '0.0182', 'NDCG@5': '0.0127', 'HIT@10': '0.0266', 'NDCG@10': '0.0154', 'HIT@20': '0.0390', 'NDCG@20': '0.0185'}
  ```

# Acknowledgement
 - Training pipeline is implemented based on [CoSeRec](https://github.com/YChen1993/CoSeRec).
 - SASRec model are implemented based on [RecBole](https://github.com/RUCAIBox/RecBole). 

Thanks them for providing efficient implementation.

# Reference

Please cite our paper if you use this code.
```
@inproceedings{dang2023uniform,
  title={Uniform sequence better: Time interval aware data augmentation for sequential recommendation},
  author={Dang, Yizhou and Yang, Enneng and Guo, Guibing and Jiang, Linying and Wang, Xingwei and Xu, Xiaoxiao and Sun, Qinghui and Liu, Hong},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={37},
  number={4},
  pages={4225--4232},
  year={2023}
}
```
