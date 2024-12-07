import polars as pl
import numpy as np

df = pl.DataFrame(
    {"nrs": [1, 2, 3, None, 5],
     "names": ["foo", "ham", "spam", "egg", "None"],
     "random": np.random.rand(5),
     "groups": ["A", "A", "B", "C", "B"]}
)
print(df)
