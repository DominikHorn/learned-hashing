import plotly.express as px
import pandas as pd
import sys
import os

if len(sys.argv) < 2:
    print("Please specify the csv files to plot")
    exit(-1)

for ds in sys.argv[1:]:
    # read and prepare data
    df = pd.read_csv(ds)

    fig = px.line(df, x="bucket_lower", y="bucket_value", title=f"{ds} ({len(df)} buckets)")
    fig.write_image(f"{os.path.splitext(ds)[0]}.png", scale=2)

    #df['bucket_value_sum'] = df['bucket_value'].expanding().sum()
    #df['bucket_value_sum'] /= df['bucket_value_sum'].max()
    #fig = px.line(df, x="bucket_lower", y="bucket_value_sum", title=f"{ds} ({len(df)} buckets)")
    #fig.show()
