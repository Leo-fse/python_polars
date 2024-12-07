import polars as pl
import numpy as np
import os
from pathlib import Path

root_dir = Path(os.getcwd())
data_dir = root_dir / "data"

df = pl.DataFrame(
    {"nrs": [1, 2, 3, None, 5],
     "names": ["foo", "ham", "spam", "egg", "None"],
     "random": np.random.rand(5),
     "groups": ["A", "A", "B", "C", "B"]}
)
print(df)


# df.select(・・・) 列の抽出
df_names = df.select(pl.col("names"))
print(df_names)

# nrs列にrandom列の平均値と10を足す。 自動的にブロードキャストされる。
df_result = df.select(
    pl.col("nrs") + pl.col("random").mean() + 10
)
print(df_result)

# 複数のエクスプレッションを式に渡す
df_result = df.select(
    pl.col("names"),
    pl.col("nrs") * 2,
    pl.col("groups")
)
print(df_result)

# df.with_columns(・・・) データフレームすべてを返す+新しい列を追加
df_result = df.with_columns(
    (pl.col("nrs") + pl.col("random").mean() + 10).alias("feature")
) 
print(df_result)

# df.filter(・・・) データフレームのフィルタリング
df_result = df.select(
    pl.col("groups") == "A"
)
print(df_result)

df_filterd = df.filter(
    pl.col("groups") == "A"
)
print(df_filterd)

# フィルターへの複数条件の指定
df_filterd = df.filter(
    (pl.col("nrs") > 1) 
    & (pl.col("groups") == "A")
)
print(df_filterd)

# filter式に複数のエクスプレッションを渡した場合は、それらはANDで結合されます。
df_filterd = df.filter(
    pl.col("nrs") > 1,
    pl.col("groups") == "A"
)
print(df_filterd)

# データの表示
fp =  data_dir / "titanic" / "train.csv"
if not fp.exists():
    raise FileNotFoundError(f"{fp} not found.")
df = pl.read_csv(fp, encoding="cp932")

print(df.head(3))

# データ表示行数の指定
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)

print(df.head(100))

# データフレームの内容をざっと表示
print(df.glimpse())

# データの列名を確認
print(df.columns)

# データの大きさの確認
print(df.shape, df.height, df.width)

# 各列のデータ型を確認
print(df.schema)

# データの概要を確認
print(df.describe())

# 欠損値のカウント
print(df.null_count())

# 列ごとのユニークな要素数の確認
print(df.select(pl.all().n_unique()))

# 重複していないユニークなレコード数の確認
print(df.n_unique())

# 列ごとのユニークな要素の表示
print(df.select(pl.col("Embarked").unique()))

# ユニークな要素ごとに重複数のカウント
result = ( 
    df
    .select(pl.col("Embarked"))
    .to_series()
    .value_counts(sort=True, normalize=True)
)
print(result)

# 特定の列に対する統計量の確認
result = df.select(
    pl.col("Age").max().alias("Age_max")
)
print(result)

result = df.select(
    pl.max("Age").alias("Age_max"),
    pl.min("Age").alias("Age_min"),
    pl.mean("Age").alias("Age_mean"),
    pl.std("Age").alias("Age_std"),
    pl.quantile("Age", 0.25).alias("Age_quantile_25"),
)
print(result)

# データフレームのソート
result = df.sort(["Fare", "Age"], nulls_last=True, descending=[True, False])
print(result)

# データフレームの複製
df_tempolary = df.clone()
print(df_tempolary)

# データフレームから複数列を抽出
result = df.select(
    pl.col(["Age", "Fare", "Embarked"])
)
print(result.head(3))

result = df.select(
    pl.col("Age"),
    pl.col("Fare"),
    pl.col("Embarked")
)
print(result.head(3))

# データフレームから特定データ型の列抽出
df_numeric = df.select(
    pl.col(pl.Int64),
    pl.col(pl.Float64)
)
print(df_numeric.head(3))

df_numeric = df.select(
    pl.col([pl.Int64, pl.Float64])
)
print(df_numeric.head(3))

# データフレームから特定の列を除外
result = df.select(
    pl.all().exclude(["Name", "Ticket"])
)
print(result.head(3))

result = df.select(
    pl.all().exclude([pl.Int64, pl.Float64])
)
print(result.head(3))

result = df.select(
    pl.all().exclude("^.*Id$")
)
print(result.head(3))

# データフレームの特定行の抽出
result = df.slice(20, 30)
print(result)

# 特定条件にマッチする行の抽出
result = df.filter(
    pl.col("Age").eq(70) 
)
print(result)

# 複数条件にマッチする行の抽出
result = df.filter(
    (pl.col("Age") > 50) & (pl.col("Fare") > 30)
)
print(result)

# 複数条件にマッチする行の抽出２
result = df.filter(
   ~( pl.col("Age").is_between(30, 45, closed="left")
    | (pl.col("Embarked") == "Q"))
)
print(result)