import ydb
import pandas as pd
import json

from dotenv import load_dotenv
import os

load_dotenv()
endpoint = os.getenv('ENDPOINT')
database = os.getenv('DATABASE')


def run_decorator(func):

    def wrapper(*args, **kwargs):

        with ydb.Driver(endpoint=endpoint, database=database) as driver:
            driver.wait(timeout=5, fail_fast=True)
            with ydb.SessionPool(driver) as pool:
                return pool.retry_operation_sync(
                    lambda ses: func(ses, *args, **kwargs))

    return wrapper


@run_decorator
def create_table(session, name='row_data_test'):
    session.execute_scheme(f"""
                CREATE table `{name}` (
                    `key` Uint64,
                    `data` Utf8,
                    PRIMARY KEY (`key`)
                )
                """)


@run_decorator
def insert(session, name='row_data_test'):
    with open('data/result.json', 'r') as file:
        json_str_list = json.load(file)
    for i, row in enumerate(json_str_list):
        query = """
        INSERT INTO {name} (key, data)
        VALUES ({key}, "{data}");
        """.format(key=i,
                   data=str(row).replace('True', 'true'),
                   name=name)
        session.transaction().execute(
            query=query,
            commit_tx=True,
        )


@run_decorator
def read_bach(session, start, size, name):
    query = f"""SELECT * FROM {name}
                ORDER BY key
                LIMIT {size}
                OFFSET {start}"""
    return session.transaction().execute(query=query)[0].rows

def read_by_baches(size=1000, name='row_data_test'):
    start = 0
    while True:
        data = read_bach(start, size, name)
        if len(data) < size or len(data) == 0:
            break
        yield data
        start += size
