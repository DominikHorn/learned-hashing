import plotly.express as px
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Please specify the csv file as first parameter")
    exit(-1)

df = pd.read_csv(sys.argv[1])
fig = px.bar(df, x="bucket_lower", y="bucket_value")
fig.show()
