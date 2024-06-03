import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
from datetime import datetime

def load_metrics(filename):
    """Loads the metrics from a specified file."""
    undesired_columns = [2, 3, 4, 5, 6, 14]
    with open(filename) as file:
        lines = list(csv.reader(file, delimiter=",", quotechar="\""))
    metrics_data = np.array(lines)
    data = np.delete(metrics_data, undesired_columns, 1)
    return data

def unstructured_to_structured(data, indexes):
    """Takes an unstructured numpy array from raw data and structures it."""
    data_no_header = data[1:]
    header = data[0].tolist()
    tuples = [tuple(row) for row in data_no_header.tolist()]
    dtype = []

    for i in range(0, 9):
        if i in indexes:
            dtype.append((header[i], "<U30"))
        else:
            dtype.append((header[i], "float"))

    structured_array = np.array(tuples, dtype=np.dtype(dtype))
    return structured_array

def convert_timestamps(array):
    """Converts raw time data into a structured consistent type."""
    list_of_time = array.tolist()
    cleaned_time = []

    for time in list_of_time:
        parts = time.split()
        parts.pop(4)
        cleaned_time.append(" ".join(parts))

    converted_time = [datetime.strptime(time, "%a %b %d %H:%M:%S %Y") for time in cleaned_time]
    return np.array(converted_time)

dropout = 0.1
intensity_columns = ["valence_intensity", "anger_intensity", "fear_intensity", "sadness_intensity", "joy_intensity"]

# Uncomment the following lines when you reach task 4.
# for column in intensity_columns:
#     data[column][np.random.choice(data[column].size, int(dropout * data[column].size), replace=False)] = np.nan

def replace_nan(data):
    """Replaces the NaN values in a numpy array with the mean of the column."""
    for column in intensity_columns:
        mean = np.nanmean(data[column], axis=0)
        data[column][np.isnan(data[column])] = mean
    return data

def plot_boxplot(data, output_name="output.png"):
    """Constructs a boxplot of intensity values."""
    data_to_plot = [data[col] for col in intensity_columns]
    labels = ["Valence", "Anger", "Fear", "Sadness", "Joy"]
    colors = ["green", "red", "purple", "blue", "yellow"]

    plt.figure(figsize=(10, 7))
    median_props = dict(linestyle="-", linewidth=1, color="black")
    plot = plt.boxplot(data_to_plot, patch_artist=True, medianprops=median_props, labels=labels)

    for patch, color in zip(plot["boxes"], colors):
        patch.set_facecolor(color)

    plt.grid(True, axis="y")
    plt.title("Distribution of Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Values")
    plt.savefig(output_name)

def count_outliers(sentiment, lower, upper):
    """Returns the number of outliers in the covid sentiment data."""
    quartile1 = np.percentile(sentiment, lower)
    quartile3 = np.percentile(sentiment, upper)
    outliers = ((sentiment >= quartile3) + (sentiment <= quartile1))
    return outliers.sum()

def convert_to_dataframe(data):
    """Converts the previous numpy array into a pandas dataframe."""
    return pd.DataFrame(data)

def load_tweets(filename):
    """Loads tweets from a TSV file."""
    return pd.read_csv(filename, sep="\t")

def merge_dataframes(df_metrics, df_tweets):
    """Merges the tweets dataframe with the covid sentiments dataframe on the tweet_ID column."""
    df_metrics = df_metrics.astype({"tweet_ID": float})
    df_metrics.set_index("tweet_ID", inplace=True)
    df_tweets.rename(columns={"id": "tweet_ID"}, inplace=True)
    df_merged = df_tweets.join(df_metrics, on="tweet_ID", how="inner", lsuffix="", rsuffix="_tweet_ID")
    df_merged = df_merged.astype({"tweet_ID": "int64"}).dropna()
    return df_merged

def plot_time_period(df_merged, from_date, to_date, output_name="output.png"):
    """Creates a series of line plots showing covid sentiment over a specified date range."""
    colors = ["green", "red", "purple", "blue", "yellow"]
    labels = intensity_columns
    df_merged["created_at"] = pd.to_datetime(df_merged["created_at"])
    df_merged.sort_values(by="created_at", inplace=True)

    date_mask = (df_merged["created_at"] > from_date) & (df_merged["created_at"] < to_date)
    data_filtered = df_merged.loc[date_mask]

    plt.figure(figsize=(15, 8))
    plt.xlabel("created_at")
    plt.xticks(rotation=30, ha="right")

    for i, label in enumerate(labels):
        plt.plot(data_filtered["created_at"], data_filtered[label], color=colors[i], label=label)

    plt.legend()
    plt.savefig(output_name)

def get_top_n_words(column, n):
    """Retrieves the top n words and their frequencies from given data."""
    frequencies = Counter()
    column.str.lower().str.split().apply(frequencies.update)
    return frequencies.most_common(n)

# Example usage:
# word_frequency = get_top_n_words(df_merged["text"], 50)

def plot_word_frequency(word_frequency, n, output_name="output.png"):
    """Creates a bar chart of word frequencies."""
    words, frequencies = zip(*word_frequency)
    y_pos = np.arange(len(words))

    fig, ax = plt.subplots(figsize=(15, 10))
    colors = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
    color_map = mcolors.LinearSegmentedColormap.from_list("", list(zip(np.linspace(0, 1, len(colors)), colors)))

    ax.barh(y_pos, frequencies, align="center", color=color_map(np.linspace(0, 1, len(words))))
    ax.set_yticks(np.arange(len(words)))
    ax.set_yticklabels(words)
    plt.gca().invert_yaxis()
    plt.xlabel("Frequency")
    plt.title(f"Word Frequency: Top {n}")
    plt.savefig(output_name)
