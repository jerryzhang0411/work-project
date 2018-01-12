使用方法：
1. 将raw data与script放在同一目录下
2. 使用cmd进入目录， 使用命令："python script.py"或"python script_huge.py"，按照提示输入raw data的文件名
P.S. 一般情况下使用"python script.py"，当raw data过大（>2G）时使用"python script_huge.py"
3. 脚本中临近结束有一段"random test"代码(Line 311-336)，会随机抽样（样本/11）数量的样本做预测并探究正确率，生成match_test_error.txt。若想做此测试将那段代码comment out掉即可直接使用

输出:
1. RF_Model: 训练出来的Random Forest模型
2. error_list.txt: 整个样本中原本与预测的标记不一样的instance报告
(仅对script.py)-
3. word_vector.txt: 用来训练的特征向量
4. 高频词词典.txt: 用来生成特征向量的高频词集

占用时间：
	对一个16302条数据的数据集约用时60分钟，请耐心等待