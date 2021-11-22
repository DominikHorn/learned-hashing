import plotly.express as px
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Please specify the csv files to plot")
    exit(-1)

for ds in sys.argv[1:]:
    df = pd.read_csv(ds)
    fig = px.line(df, x="bucket_lower", y="bucket_value", title=ds)
    fig.show()
