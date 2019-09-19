# 2019-false-news-detection-challenge
Bert chinese text classification by PyTorch

# 赛题网址：
https://www.biendata.com/competition/falsenews/

# 比赛任务：
分为三个子任务，即Task1：虚假新闻文本检测、Task2：虚假新闻图片检测和 Task3：虚假新闻多模态检测，相关任务描述详见官方网址.

# 解决方案：
我仅参加了Task1：虚假新闻文本检测，使用Bert预训练模型，训练了两个epoch后，线下有效的f1值98.71%(训练数据集句子相似度太高，划分的验证集在训练集也存在高度相似的句子)，提交线上初赛测试集的分数为88.79%，由于没有GPU没有进行调参再训练，你可以结合官方提供的辟谣文本，去除重复的或者相似的高的句子进行训练.

# 运行环境
numpy

pickle

scikit-learn

PyTorch 1.0

matplotlib

pandas

pytorch_transformers=1.1.0

# 如何使用代码
1.需要下载Bert pretrained model、config和vocab文件，下载链接：https://pan.baidu.com/s/1Kp9p-y3GhzMtp8IZaIcpnQ 提取码：97q3 

2.把model、config和cocab文件放到/pybert/pretrain/bert/base-chinese 目录中.

3.准备数据，放入到pybert\dataset目录中，数据可以到我分享的网盘下载https://pan.baidu.com/s/1N-DjhWMTnZpPzEJjv3CoBw 提取码: ethu，也可以去官方地址下载.

4.运行 python run_bert.py --do_data 对数据进行预处理.

5.运行 python run_bert.py --do_train --save_best --do_lower_case 进行模型训练.

6.运行 run_bert.py --do_test --do_lower_case 进行新数据预测.

# 训练
[Training] 1475/3848 [==========>...................] - ETA: 17:24:51  loss: 0.0004 

Epoch: 2 -  loss: 0.0363 - acc: 0.9916 - f1: 0.9916 - valid_loss: 0.0679 - valid_acc: 0.9871 - valid_f1: 0.9871 

Epoch 2: valid_f1 improved from 0.93659 to 0.98713

# Tips
1.you can modify the io.data_transformer.py to adapt your data.

2.As recommanded by Jocob in his paper https://arxiv.org/pdf/1810.04805.pdf, in fine-tuning tasks, the hyperparameters are expected to set as following: Batch_size: 8 or 16, learning_rate: 5e-5 or 2e-5 or 3e-5, num_train_epoch: 4 or 5

3.The pretrained model has a limit for the sentence of input that its length should is not larger than 512, the max position embedding dim. The data flows into the model as: Raw_data -> WordPieces -> Model. Note that the length of wordPieces is generally larger than that of raw_data, so a safe max length of raw_data is at ~128 - 256


