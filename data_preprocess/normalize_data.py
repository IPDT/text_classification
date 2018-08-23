import re
from nltk import word_tokenize
import os

# from nltk.corpus import stopwords

replacement_patterns = [(r'won\'t', 'will not'),
                        (r'can\'t', 'cannot'),
                        (r'i\'m', 'i am'),
                        (r'ain\'t', 'is not'),
                        (r'\'ll', ' will'),
                        (r'n\'t', ' not'),
                        (r'\'ve', ' have'),
                        (r'\'s', ' is'),
                        (r'\'re', ' are'),
                        (r'\'d', ' would')]


class NormalizeData(object):
    def __init__(self, content):
        self.__raw_content = content
        self.__normalize_content = self.__raw_content
        self.__patterns = [(re.compile(regex), replace) for (regex, replace) in replacement_patterns]
        self.__punctuation = '[,，。.!?:"@#$%^&*()+=_:;“”‘’]'

    def normalize_data(self):
        self.__replace_sentences()
        self.__cut_sentences_to_words()
        self.__remove_stop_word()
        return self.normalize_content

    def __remove_stop_word(self):
        # stop_words = stopwords.words('english')
        f_stop_word = open('data_preprocess/stop_word/stop_word.txt', 'r', encoding='utf-8')
        stop_word_list = f_stop_word.readlines()
        remove_stop_sentences = self.normalize_content
        for sentence in remove_stop_sentences:
            for word in sentence:
                if word in stop_word_list:
                    sentence.remove(word)
        self.normalize_content = remove_stop_sentences

    def __cut_sentences_to_words(self):
        cut_lower_sentences = []
        for sentence in self.normalize_content:
            non_punc_sentence = re.sub(self.__punctuation, ' ', sentence.replace('\n', ''))
            non_punc_sentence = re.sub(r'\s+', ' ', non_punc_sentence)
            cut_lower_sentences.append(word_tokenize(non_punc_sentence.lower()))
        self.normalize_content = cut_lower_sentences

    def __replace_sentences(self):
        replace_sentence = []
        for sentence in self.__raw_content:
            for (pattern, replace) in self.__patterns:
                (sentence, count) = re.subn(pattern, replace, sentence.lower())
            replace_sentence.append(sentence)
        self.normalize_content = replace_sentence

