# Benchmarking ML models: Finding the best Algorithm for Network Threat Prediction

## Project Overview

This project is a **systemic comparative study** aimed at identifying the most
effective ML algorithm for **Network Anomaly and Intrusion Detection (NIDS)**, based on
performance metrics

The study trains and evaluates a set of tree-based and ensemble **RandomForestClassifier**,**IsolationForest** and **XGBoostClassifier** on the **NSL-KDD dataset**.
The primary goal is to determine the model that offers the best balance of high **Recall** and **F1-Score** to minimize missed threats (False Negative) 

## Dataset
* **File:** `NSL-KDD.csv`

 ## Models 
 **RandomForestClassifier** (supervised algorithm)
 * **Evaluation:** Accuracy,Precision,Recall,F1-score
 * **Outputs:** Charts of metrics versus the percentage of the dataset used for training

**IsolationForest** (unsupervised algorithm)
* **Evaluation:** Classification_report
* **Outputs:** Charts of metrics versus the percentage of the dataset used for training

**XGBoostClassifier** (supervised algorithm)
* **Objective:** 'multi:softmax'
* **Learning Rate:** 0.3
* **n_estimators:** 500
* **random_state:** 42
* **Evaluation:** Accuracy
* **Outputs:** Charts of accuracy metric versus usage of dataset while training

## Functions
creating funtions for datas preprocess such as having numeric and encoded datas in all .csv file.
1. **drop_nan(df,pd.DataFrame)**
2. **categorize_protocol(df,collumn_name,start_num=1)**
3. **drop_train_unused(df,collumn_name)**
4. **encode_label(df,collumn_name,axis=1,inplace=True)**
5. **collumn_fillna(df)**

## Training PipeLine
1. ***Preprocessing***
  - import functions.py in order to preprocess datas and you can use various functions depend on the dataset you have
2. **RandomForestClassifier Model Training**
    - using traing_test_split in order to split dataset
    - creating train_size(array 0.15-0.60 with 0.05 pace)
    - A set of ten models was generated. Each model comprises ten sub-models, each trained using a varying fraction of the dataset (e.g., Sub-model 1 of Model 1 used 85% of        the data, Sub-model 2 used 80%, and so on).
 
 3. **IsolationForest Model Training**
     - drop labels collun because IsolationForest is an unsupervised algorithm. This algorithm does not build relationships between the data points, but rather isolates the          most 'anomalous' (or 'outlying') data.
     - using traing_test_split in order to split dataset
     - creating train_size(array 0.15-0.60 with 0.05 pace)
     - A set of ten models was generated. Each model comprises ten sub-models, each trained using a varying fraction of the dataset (e.g., Sub-model 1 of Model 1 used 85% of        the data, Sub-model 2 used 80%, and so on).
  
  4. **XGBoostClassifier Model Training**
       - Further data preprocessing was required, as we needed to find values in the 'label' column that appeared fewer than two times.
       - using RobustScaler and LalelEncoder
       - A set of ten models was generated. Each model comprises ten sub-models, each trained using a varying fraction of the dataset (e.g., Sub-model 1 of Model 1 used 85%            of the data, Sub-model 2 used 80%, and so on).
    

  ## Results
 | Model   | Accuracy | Precision | Recall | F1-Score | Notes |
|---------|----------|----------|-------|-------|-------|
| **RandomForestClassifier** | ~98.5%     |     |  | Stable training, strong generalization,Fast training |

| Model   | Accuracy |  Notes |
|---------|----------|-------|
| **XGBoostClassifier** | ~98.5-98.9%     |     Stable training, strong generalization |

| Model   | Accuracy | Macro F1 | Notes |
|---------|----------|----------|-------|
| **IsolationForest** |      |     |  |


## Requirements
- Python 3.13.2
- NumPy
- Pandas
- XGBoost
- scikit-learn
- Matplotlib
- SeaBorn
