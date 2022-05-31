import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np


df = pd.read_csv("report_mcp_traffic_accidents.csv")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print(df.to_string())
