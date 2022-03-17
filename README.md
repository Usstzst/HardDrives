# HardDriveAnalysis

## 1. 需求

复现 https://www.kaggle.com/awant08/hard-drive-failure-prediction-st4000dm000

1. 给出 ST4000DM000 TRAIN DATASET: from 2015 to 2020 TEST DATASET: 2021 下的5个model 的 FAR和 FDR
(形式参考  (https://github.com/awant/sd_failure_predictions#results ) )
2. (可选)给出 DenseNet+LSTM model 的 FAR和 FDR

建议将复现的结果提交到个人的 gitlab 仓库中,面试时提供链接，方便讨论。



## 2. 数据

Each day in the Backblaze data center, we take a snapshot of each operational hard drive. This snapshot includes basic drive information along with the S.M.A.R.T. statistics reported by that drive. The daily snapshot of one drive is one record or row of data. All of the drive snapshots for a given day are collected into a file consisting of a row for each active hard drive. The format of this file is a "csv" (Comma Separated Values) file. Each day this file is named in the format YYYY-MM-DD.csv, for example, 2013-04-10.csv.

The first row of the each file contains the column names, the remaining rows are the actual data. The columns are as follows:

- Date – The date of the file in yyyy-mm-dd format.
- Serial Number – The manufacturer-assigned serial number of the drive.
- Model – The manufacturer-assigned model number of the drive.
- Capacity – The drive capacity in bytes.
- Failure – Contains a “0” if the drive is OK. Contains a “1” if this is the last day the drive was operational before failing.
- 2013-2014 SMART Stats – 80 columns of data, that are the Raw and Normalized values for 40 different SMART stats as reported by the given drive. Each value is the number reported by the drive.
- 2015-2017 SMART Stats – 90 columns of data, that are the Raw and Normalized values for 45 different SMART stats as reported by the given drive. Each value is the number reported by the drive.
- 2018 (Q1) SMART Stats – 100 columns of data, that are the Raw and Normalized values for 50 different SMART stats as reported by the given drive. Each value is the number reported by the drive.
- 2018 (Q2) SMART Stats – 104 columns of data, that are the Raw and Normalized values for 52 different SMART stats as reported by the given drive. Each value is the number reported by the drive.
- 2018 (Q4) SMART Stats – 124 columns of data, that are the Raw and Normalized values for 62 different SMART stats as reported by the given drive. Each value is the number reported by the drive.



## 3. 数据预处理

* 按年处理：同一年份数据拼接

* 数据探查：缺失值、唯一值等等探查

* 训练集和测试集处理

  

## 4. 数据分析

暂定

## 5. 模型搭建

![image-20220314224151367](C:\Users\Julianna\AppData\Roaming\Typora\typora-user-images\image-20220314224151367.png)







TrainDataSet: from 2015 to 2017

TestDataSet: 2021(health, failured)

| model         | FAR   | FDR   |
| ------------- | ----- | ----- |
| KDD hardcoded | 0.094 | 0.708 |
| RNN Net       |       |       |
| Dense Net 32  |       |       |
| Dense Net 8   |       |       |
| Dense Net 8,8 |       |       |

