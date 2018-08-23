import pymysql
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
                labels.append(row['label'])
                contents.append(row['data'])
        return labels, contents
