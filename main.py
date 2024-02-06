import pandas as pd
import numpy as np
import json
import regex
from sqlalchemy import create_engine
import lib
from dotenv import load_dotenv
import os
import logging

load_dotenv()
engine = create_engine(os.getenv('POSTGRES'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def is_anon(uuid: list) -> list:
    return [True, False] * (len(uuid) // 2)


def transform(bath):
    batch2 = map(lambda x: x['data'].replace("'", '"'), bath)
    batch3 = map(json.loads, batch2)
    cols = ['uuid', 'name', 'date', 'region', 'city', 'created_at']
    df = pd.DataFrame(batch3, columns=cols)
    df['anon'] = is_anon(df['uuid'].to_list())
    # df['quality'] = 10
    df['quality_comment'] = ''

    # фильтруем тестовые значения
    mask = df['name'].str.lower().str.contains('test')
    df.loc[mask, 'quality_comment'] = 'test'

    # df.loc[0, 'name'] = df.loc[0, 'name'] + '%'
    # pattern = r'^[A-Za-zА-Яа-я0-9]+$'
    # mask = df['name'].apply(lambda x: bool(re.match(pattern, x)))
    # df.loc[mask, 'quality_comment'] = 'special symbols'

    # в имене больше 3 слов
    mask = df['name'].apply(lambda x: len(x.split()) > 3)
    df.loc[mask, 'quality_comment'] = 'more then 3 words in name'

    # приводим дату к формату даты
    date = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    mask = df['date'].notna() & date.isna()
    df.loc[mask, 'quality_comment'] = 'bad date'
    df['date'] = date

    # аномизируем часть данных
    mask = df['anon']
    df.loc[mask, 'date'] = df[mask]['date'].dt.to_period('Y').dt.start_time

    names = pd.read_csv('rendom_names.csv')
    names = names['men'].to_list() + names['women'].to_list()
    names_sample = np.random.choice(names, size=mask.sum())
    df.loc[mask, 'name'] = names_sample
    stat = df['quality_comment'].value_counts().drop('')
    s = str(stat.to_dict())[1:-1]
    logger.info(f'{stat.sum()} problemms: ' + s)
    # удаляем тест
    return df[df['quality_comment'] != 'test']


def pipeline(size, source_table_name, dist_table_name):
    for i, batch in enumerate(lib.read_by_baches(size=size)):
        logger.debug(f'read bath #{i}')
        df = transform(batch)
        logger.debug(f'transform bath #{i}')
        df.to_sql(dist_table_name, con=engine, index=False, if_exists='append')
        logger.debug(f'write bath #{i}')

if __name__ == '__main__':
    # logging.getLogger('kuku').setLevel(logging.WARNING)
    pipeline(size=10,
             source_table_name='row_data_test',
             dist_table_name='topups')
