from data_preprocess.mysql_config import MySQLConfig
from data_preprocess.normalize_data import NormalizeData
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np


class TextSim(object):
    def __init__(self, content):
        self.__content = [content]
        self.__mysql = MySQLConfig()
        self.__normalized_contents = self.__content

    def text_similarity(self, threshold=0.2):
        labels, raw_contents = self.__mysql.read_raw_data()
        contents = self.__content + raw_contents
        normalize_data = NormalizeData(contents)
        self.__normalized_contents = normalize_data.normalize_data()
        list_tf_idf = self.__calculate_tf_idf()
        print('Size of all text in tf-idf: ' + str(len(list_tf_idf)))

        vec_content = list_tf_idf[0]
        dic_cos = {}
        for i in range(1, len(list_tf_idf)):
            if i % 10000 == 0:
                print('Processing tf-idf in step ' + str(i))
            vec = list_tf_idf[i]
            cos = np.dot(vec_content, vec) / (np.linalg.norm(vec_content) * (np.linalg.norm(vec)))
            if cos > threshold:
                dic_cos[i] = cos
        cos_reverse_sort = sorted(dic_cos.items(), key=lambda x: x[1], reverse=True)
        key = list(zip(*cos_reverse_sort))[0]
        value = list(zip(*cos_reverse_sort))[1]

        text_sim_list = []
        for i in range(len(key)):
            text_sim_list.append(labels[key[i]-1] + ' | ' + contents[key[i]] + ' : ' + str(value[i]))
        print('Size of result after text similarity: ' + str(len(text_sim_list)))
        return text_sim_list

    def __calculate_tf_idf(self):
        # 将文本中的词语转换为词频矩阵
        vectorizer = CountVectorizer()
        word_list = []
        for content in self.__normalized_contents:
            sentence = ''
            for word in content:
                sentence = sentence + ' ' + word
            word_list.append(sentence)
        # 计算个词语出现的次数
        X = vectorizer.fit_transform(word_list)
        # 类调用
        transformer = TfidfTransformer()
        # 将词频矩阵X统计成TF-IDF值
        tf_idf = transformer.fit_transform(X)
        return tf_idf.toarray()


if __name__ == '__main__':
    text_sim = TextSim('The only significant knock against the Insight is the engine drone.')
    content = text_sim.text_similarity()
    f_sim = open('text_sim.txt', 'w', encoding='utf-8')
    for text in content:
        f_sim.write(text + '\n')
    print(len(content))
