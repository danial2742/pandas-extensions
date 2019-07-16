from builtins import filter
from past.builtins import basestring
import functools
from collections import OrderedDict

import pandas as pd


EXTENSION_FUNCS = {}


def extension(fn):
    EXTENSION_FUNCS[fn.__name__] = fn
    return fn


@extension
def skip_columns(df, column_names):
    """
    Returns pd.Series containing all data frame columns except the specified list
    :param df Target data frame
    :param column_names List of column names to work with
    """
    return df.drop(columns=column_names, errors='ignore')


@extension
def keep_columns(df, column_names):
    return df[column_names]


@extension
def merge_all(df_list, **kwargs):
    """
    Returns pd.DataFrame with left joining of the data frames in the list.
    :param df_list List of data frames
    :param kwargs Passed to the merge function with no change
    """
    return functools.reduce(lambda df1, df2: df1.merge(df2, **kwargs), df_list)


@extension
def flatten_json_column(df, column_name, prefix=None, skip_columns_list=None):
    """
    Returns pd.DataFrame that contains all the columns from df except the specified column,
    supplemented with the columns from the specified JSON field
    :param skip_columns_list: List of column names to skip in the results of the JSON field
    :param column_name: Name of JSON column to process
    :param df: Data frame - owner of the column
    :type prefix: basestring Will be added to each column name got from the specified column.
    """
    if prefix is None:
        prefix = column_name + '_'
    non_null_rows = df[~pd.isnull(df[column_name])]
    flattened_column_data = pd.DataFrame.from_dict(non_null_rows[column_name].to_dict()).T
    if skip_columns_list:
        flattened_column_data = flattened_column_data.drop(columns=skip_columns_list)
    flattened_column_data.columns = ['%s%s' % (prefix, c) for c in flattened_column_data.columns]
    return df.drop(columns=column_name) \
        .join(flattened_column_data)


@extension
def create_column_from_df(df, column_name, df_mapper_func):
    if not callable(df_mapper_func):
        raise ValueError('Callable is expected.')
    df[column_name] = df_mapper_func(df)
    return df


@extension
def create_column_from_value(df, column_name, value):
    df[column_name] = value
    return df


@extension
def change_column_types(df, columns_list, target_type):
    return set_column_types(df, {
        col: target_type for col in columns_list
    })


@extension
def create_column_from_row(df, column_name, row_mapper_func):
    if not callable(row_mapper_func):
        raise ValueError('Callable is expected.')
    if df.empty:
        df[column_name] = None
        return df
    df[column_name] = df.apply(row_mapper_func, axis=1)
    return df


@extension
def create_column(df, column_name, val_or_row_mapper_func):
    if callable(val_or_row_mapper_func):
        return create_column_from_row(df, column_name, val_or_row_mapper_func)
    return create_column_from_value(df, column_name, val_or_row_mapper_func)


@extension
def set_column_types(df, name_to_type_dict):
    """
    :param df: pandas DataFrame
    :param name_to_type_dict: column name to column type dictionary
    :return: pandas DataFrame with column types set
    """
    return df.astype(name_to_type_dict)


@extension
def safe_sum_by_index(df, index):
    if index in df.index:
        return df.loc[index, :].sum()
    return 0


@extension
def add_missing_indices(df, expected_indices, value_or_func=None):
    """
    Ensures that rows with all expected indices are present in the data frame.
    :param df pandas DataFrame
    :param expected_indices list of indices to check
    :param value_or_func value to fill in the missing indices or function which will be executed for each missing index.
                         The function should return either one value or a set of values
    """
    missing_indices = (index for index in expected_indices if index not in df.index)
    set_value = value_or_func if callable(value_or_func) else lambda index: value_or_func
    for missing_index in missing_indices:
        df.loc[missing_index, :] = set_value(missing_index)
    return df


@extension
def add_missing_columns(df, expected_columns, value=None, types_dict=None, default_type=None):
    """
    Ensures that all expected columns are present in the data frame.
    """
    if types_dict is None:
        types_dict = {}
    missing_columns = (col for col in expected_columns if col not in df.columns)
    for missing_column in missing_columns:
        df[missing_column] = value
        if missing_column in types_dict:
            target_type = types_dict[missing_column]
            df[missing_column] = df[missing_column].astype(target_type)
        elif default_type is not None:
            df[missing_column] = df[missing_column].astype(default_type)
    return df


@extension
def filter_columns(df, columns_predicate):
    """
    Returns pd.Series containing only columns that matched the columns_predicate.
    """
    if isinstance(columns_predicate, basestring):
        columns = [col for col in df.columns if columns_predicate in col]
    else:
        columns = list(filter(columns_predicate, df.columns))
    return df[columns]


@extension
def row_count(df):
    return df.shape[0]


@extension
def column_count(df):
    return df.shape[1]


@extension
def map_rows(df, mapper_func):
    return df.apply(mapper_func, axis=1)


# TODO: remove
def workaround_15106(df, original_df, index_columns):
    """
    workaround for https://github.com/pandas-dev/pandas/issues/15106
    Adds missing columns from the original data frame in case current data frame is empty.
    Use when grouping an empty data frame by multiple columns.
    """
    if df.empty:
        df.add_missing_columns(original_df.columns.tolist() + index_columns)
        return df.set_index(index_columns)
    return df


@extension
def to_ordered_dict(df):
    result = OrderedDict()
    for _, row in df.iterrows():
        result[row.name] = OrderedDict(row)
    return result
