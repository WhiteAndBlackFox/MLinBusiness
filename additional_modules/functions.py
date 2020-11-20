# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def count_is_null_and_duplicate(df: pd.DataFrame):
    """
    Подсчитаем количество пустых значений и дубликатовы
    :param df: Dataframe данных
    """

    print("Пустые значения!")
    na = df.isnull().sum()
    print(na)

    print("-" * 30)
    dup = df.duplicated().sum()
    print(f"Количество дубликатов: {dup}")


def view_null_values(df: pd.DataFrame, name_column: str):
    na = df[df[name_column].isnull()]
    print(na)

    print("-" * 30)
    print("Список индексов пустых:")

    na_list = df.index[df[name_column].isnull()].tolist()
    print(na_list)

    return na_list


def view_duplicated_values(df: pd.DataFrame, name_column: str):

    dup = df[df[name_column].duplicated()]
    print(dup)

    print("Список индексов дупликатов")

    dup_list = df.index[df[name_column].duplicated()].tolist()
    print(dup_list)

    return dup_list


def remove_nan_or_dup_values(df: pd.DataFrame, values_list: list):

    for na in values_list:
        idx_loc = df.index.get_loc(na)

        idx_pre = 1 if idx_loc == 0 else -1
        idx_next = -1 if idx_loc == len(df) else (1 if idx_pre < 0 else 2)

        df.iloc[idx_loc, :] = (df.iloc[idx_loc + idx_pre, :] + df.iloc[idx_loc + idx_next, :]) / 2

    return df


def split_data(df, split_date):
    return df.loc[df.index.get_level_values(df.index.name) <= split_date].copy(), \
           df.loc[df.index.get_level_values(df.index.name) > split_date].copy()


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def report(results, n_top = 3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")