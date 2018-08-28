import re
from nltk import word_tokenize
import os

# from nltk.corpus import stopwords

replacement_patterns = [(r'won\'t|won\’t', 'will not'),
                        (r'can\'t|can\’t', 'cannot'),
                        (r'i\'m|i\’m', 'i am'),
                        (r'ain\'t|ain\’t', 'is not'),
                        (r'\'ll|\’ll', ' will'),
                        (r'n\'t|n\’t', ' not'),
                        (r'\'ve|\’ve', ' have'),
                        (r'\'s|\’s', ' is'),
                        (r'\'re|\’re', ' are'),
                        (r'\'d|\’d', ' would')]


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
        stop_word_list = f_stop_word.read().splitlines()
        remove_stop_sentences = self.normalize_content
        for sentence in remove_stop_sentences:
            i = 0
            while i < len(sentence):
                if sentence[i] in stop_word_list:
                    sentence.remove(sentence[i])
                    i = i - 1 if i > 1 else 0
                else:
                    i += 1
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


# if __name__ == '__main__':
#     print(os.getcwd())
#     raw_content = [
#         "it a The 2008 Lexus ES 350's appearance appeals to many, but some reviewers found its lines a little stodgy.Though it shares running gear with the Toyota Camry, Edmunds points out that “the ES doesn't share a single dash panel or material with its less expensive Toyota sibling.”The shape is vastly improved over the previous entry-level sedan from Lexus; 2008’s ES four-door has “a classy, conservative design,” Cars.com says, while acknowledging its “sharper angles and more defined body panel creases.” Automobile agrees--the ES 350’s “all-new suit of sheetmetal is considerably more flattering than the ES330's bathtublike shape”—as does Motor Trend, which reports that “from the nose on, the sheetmetal flows into a rakish, aerodynamic body with a bit of the GS and IS sport sedans' flavor.” While Mother Proof describes the exterior of the 2008 Lexus ES 350 as \"curvy and sharp,\” Kelley Blue Book wonders if it may be \"a bit too conservative.\"Reviewers had few complaints about the interior.",
#         "Volvo's pretttttttty loud and coarse.",
#         "The Soul Exclaim rides a little stiffer than other small crossovers (blame the big wheels), but it's far from being a deal-breaker.",
#         "Even on long drives, you're unlikely to feel fatigued.",
#         "Active noise canceling reduces extraneous wind, road and powertrain noise.",
#         "Automatic three-zone climate control is standard.",
#         "And all trims aside from the base benefit from an air ionizer.",
#         "The available adaptive suspension should further help smooth out the ride."]
#     test = NormalizeData(raw_content)
#     print(test.normalize_data())
