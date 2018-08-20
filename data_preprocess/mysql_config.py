import pymysql
import re
from random import shuffle


class MySQLConfig(object):
    def __init__(self, language_type='english'):
        self.language_type = language_type
        schema = language_type == 'chinese' and 'ml_features' or 'ml_features_en'
        print('Connecting Database Schema : ' + schema)
        self.connection = pymysql.connect(host='10.58.0.189', user='I339493', password='test123', db=schema,
                                          charset='utf8',
                                          cursorclass=pymysql.cursors.DictCursor)

    def read_raw_data(self):
        labels, contents = [], []
        with self.connection.cursor() as cursor:
            cursor.execute('select label,data from RAW_DATA order by rand()')
            result_set = cursor.fetchall()
            shuffle(result_set)
            for row in result_set:
                labels.append(self.native_content(row['label']))
                if self.language_type == 'chinese':
                    contents.append(list(self.native_content(row['data'])))
                else:
                    contents.append(list(self.native_content(row['data']).split()))
        return labels, contents

    @staticmethod
    def native_content(content):
        punc = '[.!\'?:"@#$%^&*()+=_:;“”‘’]'
        no_punc_content = re.sub(punc, ' ', content.replace('\n', ''))
        return re.sub(r'\s+', ' ', no_punc_content)
