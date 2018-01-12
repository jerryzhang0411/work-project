from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import sklearn.cross_validation as cv
import numpy as NP
import jieba
import operator
from collections import Counter
import linecache
import random
import warnings
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
} # dictionary for the label: [data[11]:label]

frequencyDict = {}

print("请输入数据文件名：")
filename = input()

try:
	open(filename, "r", encoding='UTF-8')
except:
	print("找不到该文件")
else:
	with open(filename, "r", encoding='UTF-8') as file:
		for line in file:
			temp_array = line.strip().split(",") # array of original sentence separate with comma

			first = '' # first sentence to be processed
			if(temp_array[47] != 'null'):
				first = temp_array[47]
			else:
				if(temp_array[48] != 'null'):
					first = temp_array[48]
				else:
					if(temp_array[49] != 'null'):
						first = temp_array[49]
					else:
						if temp_array[-37] != 'null':
							second = temp_array[-37]
							seg_list = jieba.cut(second, cut_all = False)
							for i in seg_list:
								if i not in frequencyDict:
									frequencyDict.update({i:1})
								else:
									frequencyDict[i] += 1
						continue

			# update the frequency dictionary base on the first and second element
			seg_list = jieba.cut(first, cut_all = False)
			for i in seg_list:
				if i not in frequencyDict:
					#if i not in stopList:
					frequencyDict.update({i:1})
				else:
					#if i not in stopList:
					frequencyDict[i] += 1

			if temp_array[-37] != 'null':
				second = temp_array[-37]
				seg_list = jieba.cut(second, cut_all = False)
				for i in seg_list:
					if i not in frequencyDict:
						#if i not in stopList:
						frequencyDict.update({i:1})
					else:
						#if i not in stopList:
						frequencyDict[i] += 1

	file.close()

	sorted_frequencyDict = sorted(frequencyDict.items(), key=operator.itemgetter(1), reverse = True)
	# remove the meaningless words
	del(sorted_frequencyDict[:8])

	# filter out words which have frequency smaller than 5
	true_frequencyDict = {}
	for i in sorted_frequencyDict:
		if i[1] < 5:
			break
		else:
			true_frequencyDict.update({i[0]:i[1]})




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

	with open("word_vector.txt", "w", encoding='UTF-8') as word_file:
		word_file.seek(0)
		word_file.truncate()
		with open("job_title.txt", "w", encoding='UTF-8') as title_file:
			title_file.seek(0)
			title_file.truncate()
			with open(filename, "r", encoding='UTF-8') as file:
				with open("corresponding_id.txt", "w", encoding='UTF-8') as cor_file:
					with open("corresponding_jd.txt", "w", encoding='UTF-8') as jd_file:
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
							word_file.write(''.join(str(e) for e in word_vector) + '\n') # writing word vector into a txt
							title_file.write(myDict[temp_array[11]] +"\n") # writing corresponding title into a txt
							cor_file.write(temp_array[0] + '\n')
							jd_file.write(job_description + '\n')

	with open("word_vector.txt", "r", encoding='UTF-8') as word_file:
		word_file.seek(0)
		with open("job_title.txt", "r", encoding='UTF-8') as title_file:
			title_file.seek(0)
			nb = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2)
			x_train = []
			y_train = []
			for word_line in word_file:
				title_line = title_file.readline().strip('\n')
				if title_line == 'null':
					continue
				x_train.append(NP.array(word_line.split(), dtype=NP.uint8))
				y_train.append(finalDict_P[title_line])

			nb.fit(x_train, y_train) # constructing the NB model

			word_file.seek(0)
			with open("predicted.txt", "w", encoding='UTF-8') as predict_file:
				predict_file.seek(0)

				for word_line in word_file:
					z = nb.predict(NP.array(word_line.split(), dtype=NP.uint8))
					predict_file.write(finalDict_N[z[0]]) # predict the title according to word vector
					predict_file.write('\n')

	# print("NB prediction accuracy = {0:5.1f}%".format(100.0 * nb.score(x_test, y_test)))

# iteratively train the model based on the correct value from the first time

# finding indexs that actual jobtitle and predicted jobtitle matches, will form a match.txt
i = 1
with open("job_title.txt", "r", encoding='UTF-8') as job_file:
	job_file.seek(0)
	with open("predicted.txt", "r", encoding = 'UTF-8') as predicted_file:
		predicted_file.seek(0)
		with open("match.txt", "w", encoding = 'UTF-8') as match_file:
			match_file.seek(0)
			match_file.truncate()
			for title_line in job_file:
				predicted_line = predicted_file.readline().strip('\n')
				if(predicted_line[0] == title_line[0]):
					match_file.write(str(i) + '\n')
				i+=1

total_count = 3
while total_count != 0:
	total_count -=1
	with open("match.txt", "r", encoding = 'UTF-8') as match_file:
		match_file.seek(0)
		with open("word_vector.txt", "r", encoding='UTF-8') as word_file:
			word_file.seek(0)
			with open("job_title.txt", "r", encoding='UTF-8') as title_file:
				title_file.seek(0)
				nb = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2)

				x_train = []
				y_train = []

				for match_line in match_file:
					match_line = match_line.strip('\n')

					word_line = linecache.getline("word_vector.txt", int(match_line)).strip('\n')
					title_line = linecache.getline("job_title.txt", int(match_line)).strip('\n')
					if title_line == 'null':
						continue

					x_train.append(NP.array(word_line.split(), dtype=NP.uint8))
					y_train.append(finalDict_P[title_line])

				nb.fit(x_train, y_train) # constructing the NB model with the matching word_vector and corresponding job_title

				word_file.seek(0)
				# temporary file that contain the line numbers which job title and predicted match
				with open("temp_match.txt", "w", encoding='UTF-8') as temp_match_file:
					temp_match_file.seek(0)
					with open("match.txt", "r", encoding = 'UTF-8') as match_file:
						# make prediction based on existing match.txt
						for match_line in match_file:
							word_line = linecache.getline("word_vector.txt", int(match_line)).strip('\n')
							z = nb.predict(NP.array(word_line.split(), dtype=NP.uint8))
							if(finalDict_N[z[0]][0] == linecache.getline("job_title.txt", int(match_line)).strip('\n')[0]): # predict the title according to word vector, if match, write to txt
								temp_match_file.write(match_line)

				match_file.close()
				# copy from temp to match.txt
				with open("match.txt", "w", encoding = 'UTF-8') as match_file:
					match_file.seek(0)
					match_file.truncate()
					with open("temp_match.txt", "r", encoding='UTF-8') as temp_match_file:
						for temp_match_line in temp_match_file:
							match_file.write(temp_match_line)


# re-predict based on the new model

with open("word_vector.txt", "r", encoding='UTF-8') as word_file:
	word_file.seek(0)
	with open("job_title.txt", "r", encoding='UTF-8') as title_file:
		with open("predicted.txt", "w", encoding='UTF-8') as predict_file:
			predict_file.seek(0)
			title_file.seek(0)
			

			for word_line in word_file:
				z = nb.predict(NP.array(word_line.split(), dtype=NP.uint8))
				predict_file.write(finalDict_N[z[0]]) # predict the title according to word vector
				predict_file.write('\n')

			# finding the matched items
i = 0
whole_length = 0
with open("match.txt", "w", encoding = 'UTF-8') as match_file:
	match_file.seek(0)
	match_file.truncate()
	with open("job_title.txt", "r", encoding='UTF-8') as title_file:
		whole_length = len(title_file.readlines())
		with open("predicted.txt", "r", encoding='UTF-8') as predict_file:
			predict_file.seek(0)
			title_file.seek(0)
			for title_line in title_file:
				predicted_line = predict_file.readline().strip('\n')
				if(predicted_line[0] == title_line[0]):
					match_file.write(str(i) + '\n')
				i+=1

with open("match.txt", "r", encoding = 'UTF-8') as match_file:
	with open("error_list.txt", "w", encoding = 'UTF-8') as error_file:
		original_set = set()
		for match_line in match_file:
			original_set.add(int(match_line.strip('\n')))

		full_set = set(range(1, whole_length))
		diff_set = full_set - original_set

		for i in diff_set:
			error_file.write("ID:" + linecache.getline("corresponding_id.txt", int(i)).strip('\n') + "; " + "Original Job Title is:" + linecache.getline("job_title.txt", int(i)).strip('\n') + "; " + "Predicted Job Title is:" + finalDict_N[nb.predict(NP.array(linecache.getline("word_vector.txt", int(i)).strip('\n').split(), dtype=NP.uint8))[0]]+ "; Corresponding Job Description is: "+ linecache.getline("corresponding_jd.txt", int(i)).strip('\n') + '\n')

with open('RF_Model', 'wb') as f:
    pickle.dump(nb, f)

print("Mission Accomplished")

# uncomment when you want to do "random 1500 test, output: match_test_error.txt"
# # pick random 1500 and predict again
# test_list = random.sample(range(1,15052),1500)
# with open("word_vector.txt", "r", encoding = 'UTF-8') as word_file:
# 	word_file.seek(0)
# 	with open("job_title.txt", "r", encoding = 'UTF-8') as title_file:
# 		with open("testing.txt", "w", encoding = 'UTF-8') as test_file:
# 			test_file.seek(0)
# 			title_file.seek(0)
# 			for i in test_list:
# 				z = nb.predict(NP.array(linecache.getline("word_vector.txt", int(i)).strip('\n').split(), dtype=NP.uint8))
# 				test_file.write(finalDict_N[z[0]]) # predict the title according to word vector
# 				test_file.write('\n')

# match_list = []
# error_list = set()
# with open("match_test_error.txt", "w", encoding = 'UTF-8') as match_file:
# 	match_file.seek(0)
# 	match_file.truncate()
# 	with open("job_title.txt", "r", encoding='UTF-8') as title_file:
# 		with open("testing.txt", "r", encoding='UTF-8') as predict_file:
# 			predict_file.seek(0)
# 			title_file.seek(0)
# 			for i in test_list:
# 				predicted_line = predict_file.readline().strip('\n')
# 				title_line = linecache.getline("job_title.txt", int(i)).strip('\n')
# 				if(predicted_line[0] == title_line[0]):
# 					match_list.append(i)
# 	error_list = set(test_list) - set(match_list)
# 	for i in error_list:
# 		match_file.write("ID:" + linecache.getline("corresponding_id.txt", int(i)).strip('\n') + "; " + "Original Job Title is:" + linecache.getline("job_title.txt", int(i)).strip('\n') + "; " + "Predicted Job Title is:" + finalDict_N[nb.predict(NP.array(linecache.getline("word_vector.txt", int(i)).strip('\n').split(), dtype=NP.uint8))[0]]+ "; Corresponding Job Description is: "+ linecache.getline("corresponding_jd.txt", int(i)).strip('\n') + '\n')


