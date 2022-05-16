import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import plydata as ply
import tqdm

from detecta import detect_cusum

parser = argparse.ArgumentParser(
    description="Perform changepoint detection for all tokens."
)

parser.add_argument(
    "--input_folder",
    help="The folder that contains the novel distance calculations across the years.",
)
parser.add_argument("--output_file", help="The file to output the detected changes.")

args = parser.parse_args()
final_distance_folder = args.input_folder
output_filename = args.output_file

token_time_series_dfs = {
    year_pair_file.stem.split("_")[1]: pd.read_csv(year_pair_file, sep="\t")
    for year_pair_file in Path(final_distance_folder).rglob("*tsv")
}

token_time_series = dict()
for timepoint in sorted(list(token_time_series_dfs.keys())):
    changepoint_time_series = (
        token_time_series_dfs[timepoint]
        >> ply.call(".fillna", "")
        >> ply.define(change_metric="adjusted_distance+frequency_ratio")
        >> ply.pull(["tok", "change_metric"])
    )
    for row in tqdm.tqdm(changepoint_time_series):
        token = row[0]

        if token not in token_time_series:
            token_time_series[token] = dict()
            token_time_series[token]["metric"] = list()
            token_time_series[token]["year"] = list()

        token_time_series[token]["metric"].append(row[1])
        token_time_series[token]["year"].append(timepoint)

metric_stats = list(
    itertools.chain.from_iterable(
        list(map(lambda x: x[1]["metric"], token_time_series.items()))
    )
)

# Take the 99 percentile to lower fp rate
threshold = np.percentile(metric_stats, 99)
print(f"The number of values to determine cutoff: {len(metric_stats)}")
print(f"Using the following cutoff: {threshold}")

change_point_results = []
for tok in tqdm.tqdm(token_time_series):

    alarm, start, end, amp = detect_cusum(
        token_time_series[tok]["metric"],
        threshold=threshold,
        drift=0,
        ending=True,
        show=False,
    )

    year_series = token_time_series[tok]["year"]
    for values in zip(alarm, start, end, amp):

        change_point_results.append(
            {
                "tok": tok,
                "changepoint_idx": year_series[values[0]],
                "start_idx": year_series[values[1]],
                "end_idx": year_series[values[2]],
                "value": values[3],
            }
        )

predicted_changepoints_df = (
    pd.DataFrame.from_records(change_point_results)
    >> ply.query("value > 0")
    >> ply.arrange("-value")
)
print(f"The changepoint df shape: {predicted_changepoints_df.shape}")
predicted_changepoints_df >> ply.call("to_csv", output_filename, sep="\t", index=False)
