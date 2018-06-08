# 阿里云美年大健康双高风险预测的解决方案：初赛第一，复赛14

> 我进入这个比赛比较晚，五月一号左右，之后一路和队友们杀到第一的位置。无奈复赛的时候遇到了小数据（对，就是阿里云集群上跑3000个数据。。。），最终取得了复赛第十四名。

## 跑数据之前，请注意
* 开源的是0.0279的lgb单模型，我们最终初赛第一提交的是多模型融合，这个单模型应该能排到前20名；
* 第一波针对四种不同类型的体检结果构造特征，分别是：文本型、数值型、枚举型、复合型；
* 后面又加了两波特征，组合特征能起作用的；

## 运行说明
一、 从[官网](https://tianchi.aliyun.com/competition/information.htm?spm=5176.11165320.5678.2.6e832df5OgEC40&raceId=231654)下载四个数据文件放入data文件夹
* meinian_round1_data_part1_20180408.zip
* meinian_round1_data_part2_20180408.zip
* meinian_round1_test_b_20180505.csv
* meinian_round1_train_20180408.csv

二、运行 ./src/main.py


