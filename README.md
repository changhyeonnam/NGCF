# Neural Graph  Collaborative Filtering with MovieLens in torch

## Dataset

This repository is about Neural Graph  Collaborative Filtering with MovieLens in torch. Dataset is Implict Feedback, If there is interaction between user and item, then target value will be 1.  So if there is rating value between user and movie, then target value is 1, otherwise 0. For negative sampling, ratio between positive feedback and negative feedback is 1:4 in trainset, and 1:99 in testset. (these ratios are same as [NCF](https://github.com/changhyeonnam/NCF) setting)

## Result

I measured NDCG@10 and HitRatio@10 while changing the number of embedding layers for MovieLens dataset 100k and 1M.

| dataset | Best NDCG@10 | HR@10 | # layers | epoch | batch size |
| --- | --- | --- | --- | --- | --- |
| MovieLens100k | 0.5784 | 0.8164 | 3 | 20 | 256 |
| MovieLens100k | 0.5640 | 0.8262 | 4 | 20 | 256 |
| MovieLens100k | 0.5546 | 0.8377 | 5 | 20 | 256 |
| MovieLens1m | 0.4964 | 0.7568 | 3 | 20 | 256 |
| MovieLens1m | 0.4922 | 0.7614 | 4 | 20 | 256 |
| MovieLens1m | 0.4849 | 0.7507 | 5 | 20 | 256 |

## Dependency

```java
pytorch >= 1.12.0
python >= 3.8
scipy >= 1.7.1
numpy >= 1.20.3
```

## Quick Start

```java
python3 main.py -e 10 -b 256 -dl true -k 10 -fi 100k
```

## Reference

1. [Neural Graph  Collaborative Filtering](https://arxiv.org/abs/1905.08108)
2. [Official code from Xiang Wang](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)
    
    ```java
    @inproceedings{NGCF19,
      author    = {Xiang Wang and
                   Xiangnan He and
                   Meng Wang and
                   Fuli Feng and
                   Tat{-}Seng Chua},
      title     = {Neural Graph Collaborative Filtering},
      booktitle = {Proceedings of the 42nd International {ACM} {SIGIR} Conference on
                   Research and Development in Information Retrieval, {SIGIR} 2019, Paris,
                   France, July 21-25, 2019.},
      pages     = {165--174},
      year      = {2019},
    }
    ```
