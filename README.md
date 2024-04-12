# InterKG

This is the code for the paper:
>Relation Beyond Relation: Interrelationship Learning for Knowledge Graph-based Recommendation

## 1. Introduction
InterKG enhance the connection strength between potentially associated items by extracting the interrelationship among each pair of relations. Firstly InterKG introduces relation-entity guided clustering to categorize items into different clusters, ensuring that items within each cluster share similarentity connections. Next, an indistinguishability aware relation-pair selector is proposed to choose relation pairs that are more relevant to the preferences of target users. Furthermore, an interrelationship generator is introduced to produce a virtual entity connecting potential associated items through interrelationships.

## 2. Environment Requirement
The code has been tested running under Python 3.7.10. The required packages are as follows:
```
* torch == 1.6.0
* numpy == 1.21.4
* pandas == 1.3.5
* scipy == 1.5.2
* tqdm == 4.62.3
* scikit-learn == 1.0.1
```

## 3. Run the Codes
### 3.1 pretreatment
1. run pivot_table.py to get the decision table

### 3.2 Relation-Entity Guided Clustering
1. run REG-cluster.py to get the clusters of items (In this step,we use the parameters pretrained by TransR as input, with the file path of "/InterKG/dataset/last-fm/params.pt".You can also use the training results of other models,please replace the "params.pt")

### 3.3 InterKG
1. run Selector.py to chose relations (In this step,we can get D_seg.json and p1_seg_dict.json.)
2. run Generator.py to get the new relations and entities 
3. run new-KG.py to get the new final Knowledge Graph


## 4. Datasets 
We provided two datasets to validate InterKG: last-fm and amazon-book, which are obtained from KGAT. The following table shows the information of two datasets:

|              | Last-FM | Amazon-book |
|:------------:|:-------:|:-----------:|
|    users     |  23566  |    70679    |
|    items     |  48123  |    24915    |
| interactions | 3034796 |   847733    |
|   entities   |  58266  |    88572    |
|  relations   |    9    |     39      |
|   triples    | 464567  |   2557746   |

