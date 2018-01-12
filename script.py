from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn.cross_validation as cv
import numpy as NP
import jieba
import operator
from collections import Counter
import linecache
import random
import warnings
import string
import sys
from operator import itemgetter
import pickle

warnings.filterwarnings('ignore')

myDict = {
	'技术实习-数据中心系统': 'T实习',
	'技术实习-PC客户端': 'T实习',
	'客户服务': '其他正式',
	'社招-管理支持-标准': '其他正式',
	'null': 'null',
	'技术实习-web前端/大数据云计算': 'T实习',
	'技术实习-运维研发': 'T实习',
	'实习P-商业产品': 'P实习',
	'实习-用户体验U': 'U实习',
	'技术实习-整车技术与运行管理工程师': 'T实习',
	'社招-U设计-标准': 'U正式',
	'社招-T技术-标准': 'T正式',
	'实习P-用户产品': 'P实习',
	'社招-市场及销售-标准': '其他正式',
	'技术实习-数据库': 'T实习',
	'市场及销售': '其他正式',
	'实习-管理支持': '其他实习',
	'社招-其它-标准': '其他正式',
	'技术实习-机器学习/自然语言处理/用户行为/数据挖掘': 'T实习',
	'社招-P产品-标准': 'P正式',
	'技术实习-QA': 'T实习',
	'产品': 'P正式',
	'技术实习-汽车电子测试工程师': 'T实习',
	'技术实习-后端研发': 'T实习',
	'技术实习-视觉算法/深度学习': 'T实习',
	'管理支持': '其他正式',
	'其他': '其他正式',
	'用户体验': 'U正式',
	'技术实习-移动软研': 'T实习',
	'社招-客户服务-标准': '其他正式',
	'技术': 'T正式'
} # dictionary for the label: {data[11]:label}

frequencyDict = {} # 常用词词典(未经过过滤处理)

print("请输入数据文件名：")
filename = input()

try:
	open(filename, "r", encoding='UTF-8')
except:
	sys.exit("找不到该文件")
else:
	print("已找到文件，程序运行中，请耐心等候...")
	with open(filename, "r", encoding='UTF-8') as file:
		for line in file:
			temp_array = line.strip().split(",") # array of original sentence separate with comma

			first = '' # first sentence to be processed
			second = '' # second sentence to be processed

			# temp_array[47],[48],[49]是第一段可能出现工作职责的地方
			if(temp_array[47] != 'null'):
				first = temp_array[47]
			else:
				if(temp_array[48] != 'null'):
					first = temp_array[48]
				else:
					if(temp_array[49] != 'null'):
						first = temp_array[49]
					else:
						# temp_array[-37], [-45]是第二段可能出现工作职责的地方
						if temp_array[-37] != 'null':
							second = temp_array[-37]
						else:
							if temp_array[-45] != 'null':
								second = temp_array[-45]
							else:
								continue # 两个地方都没出现相关描述

			# update the frequency dictionary base on the first and second element
			if first != '':
				seg_list = jieba.cut(first, cut_all = False)
				for i in seg_list:
					if i not in frequencyDict:
						#if i not in Dict:
						frequencyDict.update({i:1})
					else:
						#if i not in Dict:
						frequencyDict[i] += 1

			if second != '':
				seg_list = jieba.cut(second, cut_all = False)
				for i in seg_list:
					if i not in frequencyDict:
						#if i not in Dict:
						frequencyDict.update({i:1})
					else:
						#if i not in Dict:
						frequencyDict[i] += 1

	file.close()
	# 重新排序使dictionary变得有序
	sorted_frequencyDict = sorted(frequencyDict.items(), key=operator.itemgetter(1), reverse = True)
	# remove the meaningless words (top 8: 的，\\，... etc.)
	del(sorted_frequencyDict[:8])

	# filter out words which have frequency smaller than 5
	true_frequencyDict = {} # 真实高频词词典
	for i in sorted_frequencyDict:
		if i[1] < 5:
			break
		else:
			true_frequencyDict.update({i[0]:i[1]})

	with open("高频词词典.txt", "w", encoding = 'UTF-8') as freq_file:
		for i in true_frequencyDict:
			freq_file.write(str(i) + '\n')


	# 因为label不能是中文，故用数字代替
	finalDict_P = {
		'T实习': 1,
		'P实习': 2,
		'U实习': 3,
		'T正式': 4,
		'P正式': 5,
		'U正式': 6,
		'其他正式': 7,
		'其他实习': 8,
		'其他': 9
	}

	# 将数字转换为特定label
	finalDict_N = {
		1: 'T',
		2: 'P',
		3: 'U',
		4: 'T',
		5: 'P',
		6: 'U',
		7: '其他',
		8: '其他',
		9: '其他'
	}

	corresponding_id = [] # 所有instance的id，按行数排列
	corresponding_jd = [] # 所有instance的job description，按行数排列
	original_labels = [] # TUP label base on original file
	match_lines = [] # lines that original label and predicted label match
	predicted_labels = [] # predicted TUP labels base on RF model

	with open("word_vector.txt", "w", encoding='UTF-8') as word_file:
		word_file.seek(0)
		word_file.truncate()
		with open(filename, "r", encoding='UTF-8') as file:
			for line in file:
				temp_array = line.strip().split(",") # array of original sentence separate with comma
				job_description = '' # first sentence to be processed
				if(temp_array[47] != 'null'):
					job_description = temp_array[47]
				else:
					if(temp_array[48] != 'null'):
						job_description = temp_array[48]
					else:
						if(temp_array[49] != 'null'):
							job_description = temp_array[49]

				if temp_array[-37] != 'null':
					job_description += temp_array[-37]

				if temp_array[-45] != 'null':
					job_description += temp_array[-45]

				if job_description.strip(' ') == 'null' or job_description.strip(' ') == '' or len(job_description) == 1:
					continue

				word_vector = []
				job_array = jieba.cut(job_description, cut_all = False)

				counts = Counter(job_array) # counted JD

				for i in true_frequencyDict:
					if(i in counts):
						word_vector.append(str(counts[i])+' ')
					else:
						word_vector.append(str(0)+' ')
				word_file.write(''.join(str(e) for e in word_vector) + '\n') # writing word vector into a list
				original_labels.append(myDict[temp_array[11]]) # writing corresponding title into a list
				corresponding_id.append(temp_array[0]) # 根据原数据生成对应id
				corresponding_jd.append(job_description) # 对应jd

	with open("word_vector.txt", "r", encoding='UTF-8') as word_file:
		word_file.seek(0)
		rfc = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_split=2)
		# rfc = LogisticRegression(penalty= "l2")
		x_train = []
		y_train = []
		for original_labels_counter, word_line in enumerate(word_file):
			title = original_labels[original_labels_counter]
			if title == 'null':
				continue
			x_train.append(NP.array(word_line.split(), dtype=NP.uint8))
			y_train.append(finalDict_P[title])

		rfc.fit(x_train, y_train) # constructing the rfc model

		word_file.seek(0)

		for word_line in word_file:
			z = rfc.predict(NP.array(word_line.split(), dtype=NP.uint8))
			predicted_labels.append(finalDict_N[z[0]]) # predict the title according to word vector

# iteratively train the model based on the correct value from the first time

# finding indexs that actual jobtitle and predicted jobtitle matches, will form a match_lines array
for i, title in enumerate(original_labels, start=1):
	predicted = predicted_labels[i-1].strip('\n')
	if(predicted[0] == title[0]):
		match_lines.append(i)

total_count = 3
while total_count != 0:
	total_count -=1
	with open("word_vector.txt", "r", encoding='UTF-8') as word_file:
		word_file.seek(0)
		rfc = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_split=2)
		# rfc = LogisticRegression(penalty = "l2")
		x_train = [] # 单词向量
		y_train = [] # label
		
		for match_line in match_lines:
			word_line = linecache.getline("word_vector.txt", int(match_line)).strip('\n')
			title = original_labels[int(match_line)-1]
			if title == 'null':
				continue

			x_train.append(NP.array(word_line.split(), dtype=NP.uint8))
			y_train.append(finalDict_P[title])

		rfc.fit(x_train, y_train) # constructing the model with matching word_vector and corresponding label

		# temporary array that contain the line numbers which job title and predicted match
		temp_match = []
		# make prediction based on existing match
		for match_line in match_lines:
			word_line = linecache.getline("word_vector.txt", int(match_line)).strip('\n')
			z = rfc.predict(NP.array(word_line.split(), dtype=NP.uint8))
			if(finalDict_N[z[0]][0] == original_labels[int(match_line)-1][0]): # predict the title according to word vector, if match, write to array
				temp_match.append(str(match_line))

		# copy from temp to match
		match_lines = temp_match

# return tuple of (probability, ID, original_label, predicted_label, JD)
def packing(line_number):
	return (max(rfc.predict_proba(NP.array(linecache.getline("word_vector.txt", int(i)).strip('\n').split(), dtype=NP.uint8))[0])*100, corresponding_id[int(i)-1], original_labels[int(i)-1], finalDict_N[rfc.predict(NP.array(linecache.getline("word_vector.txt", int(i)).strip('\n').split(), dtype=NP.uint8))[0]], corresponding_jd[int(i)-1])

# re-predict based on the new model
with open("word_vector.txt", "r", encoding='UTF-8') as word_file:
	word_file.seek(0)
	predicted_labels.clear()
		
	for word_line in word_file:
		z = rfc.predict(NP.array(word_line.split(), dtype=NP.uint8))
		predicted_labels.append(finalDict_N[z[0]]) # predict the title according to word vector

match_lines.clear()
for i, title in enumerate(original_labels):
	predicted = predicted_labels[i].strip('\n')
	if(predicted[0] == title[0]):
		match_lines.append(str(i+1)) # the match_lines array is now 1-based

ready_queue = []
original_set = set()
for it in match_lines:
	original_set.add(int(it))

full_set = set(range(1, len(original_labels)))
diff_set = full_set - original_set # 反向选择，找出不match的行数

for i in diff_set:
	ready_queue.append(packing(i))

# sort the ready queue by the probability
ready_queue.sort(key=itemgetter(0), reverse = True)


with open("error_list.txt", "w", encoding = 'UTF-8') as error_file:
	for i in ready_queue:
		error_file.write("Probability: " + str(i[0]) + "%; ID: " + str(i[1]) + "; Original label: " + str(i[2]) + "; Predicted label:" + str(i[3]) + "; Corresponding job description: " + str(i[4]) + '\n')

with open('RF_Model', 'wb') as f:
    pickle.dump(rfc, f) # 导出训练出的模型

# # uncomment when you want to do "random test", output: match_test_error.txt

# pick random size/11 samples and predict again
testing = []
test_list = random.sample(range(1,len(original_labels)),(int)(len(original_labels)/11))
with open("word_vector.txt", "r", encoding = 'UTF-8') as word_file:
	word_file.seek(0)
	for i in test_list:
		z = rfc.predict(NP.array(linecache.getline("word_vector.txt", int(i)).strip('\n').split(), dtype=NP.uint8))
		testing.append(finalDict_N[z[0]]) # predict the title according to word vector

match_list = []
error_list = set()
final_error_list = []
with open("match_test_error.txt", "w", encoding = 'UTF-8') as match_file:
	match_file.seek(0)
	match_file.truncate()
	for idx,i in enumerate(test_list):
		predicted_line = testing[idx]
		title_line = original_labels[int(i)-1]
		if(predicted_line[0] == title_line[0]):
			match_list.append(i)
	error_list = set(test_list) - set(match_list)
	for i in error_list:
		myTuple = packing(i)
		final_error_list.append(myTuple)

	final_error_list.sort(key=itemgetter(0), reverse = True)
	for i in final_error_list:
		match_file.write("Probability: " + str(i[0]) + "%; ID: " + str(i[1]) + "; Original label: " + str(i[2]) + "; Predicted label:" + str(i[3]) + "; Corresponding job description: " + str(i[4]) + '\n')

print("Mission Accomplished")


