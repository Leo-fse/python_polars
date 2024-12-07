import pandas as pd
import polars as pl
import unicodedata
import numpy as np

def unicode_to_ascii(text):
    """Unicode文字列をASCIIに変換する関数"""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')


def decode_column_names(df):
    """DataFrameのカラム名をASCIIに変換する関数"""
    if isinstance(df, pd.DataFrame):
        if isinstance(df.columns, pd.MultiIndex):
            # マルチインデックス処理
            df.columns = pd.MultiIndex.from_tuples(
                [(unicode_to_ascii(col) if isinstance(col, str) else col) for col in multi_col]
                for multi_col in df.columns
            )
        else:
            df.columns = [unicode_to_ascii(col) for col in df.columns]
    elif isinstance(df, pl.DataFrame):
        df.columns = [unicode_to_ascii(col) for col in df.columns]
    else:
        raise TypeError('引数dfはDataFrameまたはpl.DataFrameのいずれかである必要があります')

    return df


def decode_dataframe_values(df):
    """DataFrame内の文字列データをASCIIに変換する関数"""
    if isinstance(df, pd.DataFrame):
        for col in df.columns:
            if df[col].dtype == 'O':  # オブジェクト型であれば文字列（通常はstr）
                df[col] = df[col].apply(lambda x: unicode_to_ascii(x) if isinstance(x, str) else x)
    elif isinstance(df, pl.DataFrame):
        for col in df.columns:
            if df[col].dtype == pl.Utf8:  # Polarsの文字列型
                df = df.with_columns(
                    pl.col(col).map(unicode_to_ascii)  # 追加処理を含める
                )
    else:
        raise TypeError('引数dfはDataFrameまたはpl.DataFrameのいずれかである必要があります')

    return df


def decode_dataframe(df):
    """DataFrameのカラム名と値を変換する主関数"""
    df = decode_column_names(df)
    df = decode_dataframe_values(df)
    return df
