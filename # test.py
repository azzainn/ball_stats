# test 
import pandas as pd
import numpy as np
df = pd.read_csv("stats.csv")
for column in df:
    try:
        df[column] = df[column].apply(lambda x: float(x))
    except:
        pass