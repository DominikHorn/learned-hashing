import plotly.express as px
import pandas as pd
import sys
import os
import concurrent.futures
import numpy as np

def plot(ds):
    df = pd.read_csv(ds)
    if 'models' in ds.lower():
        fig = px.line(df, x='x', y='y')
        fig.write_image(f"{os.path.splitext(ds)[0]}.png", scale=4)

    elif ds.contains('histogram'):
        fig = px.line(df, x="bucket_lower", y="bucket_value", title=f"{ds} ({len(df)} buckets)")
        fig.write_image(f"{os.path.splitext(ds)[0]}.png", scale=4)

if len(sys.argv) < 2:
    print("Please specify the csv files to plot")
    exit(-1)

executor = concurrent.futures.ProcessPoolExecutor(20)
futures = [executor.submit(plot, ds) for ds in sys.argv[1:]]
concurrent.futures.wait(futures)
