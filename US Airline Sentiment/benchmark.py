"""
benchmark.py
Time NLTK and spaCy pipelines on different sample sizes and produce a CSV summary.
Also prints tweets/sec metric.
Note: this is a simple benchmark; for robust memory profiling use memory-profiler/line_profiler.
"""

import pandas as pd
import time
import tracemalloc
import psutil
import os
from nltk_pipeline import nltk_pipeline_dataframe
from spacy_pipeline import spacy_pipeline_dataframe

def runtime_memory_wrapper(func, *args, **kwargs):
    # simple timing + memory snapshot wrapper
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    t0 = time.time()
    out = func(*args, **kwargs)
    elapsed = time.time() - t0
    mem_after = process.memory_info().rss
    return {'elapsed_sec': elapsed, 'mem_before': mem_before, 'mem_after': mem_after, 'result': out}

def run_benchmarks(path='Tweets.csv', sizes=[1000, 5000, 10000], text_col='text'):
    df_all = pd.read_csv(path)
    summary = []
    for n in sizes:
        print(f"\n--- Running benchmark for size: {n} ---")
        df = df_all.head(n)

        print("Running NLTK pipeline...")
        res_nltk = runtime_memory_wrapper(nltk_pipeline_dataframe, df, text_col, f'nltk_{n}.csv', None)
        tweets_per_sec_nltk = n / res_nltk['elapsed_sec'] if res_nltk['elapsed_sec'] > 0 else 0

        print("Running spaCy pipeline...")
        res_spacy = runtime_memory_wrapper(spacy_pipeline_dataframe, df, text_col, f'spacy_{n}.csv', f'spacy_entities_{n}.json', None)
        tweets_per_sec_spacy = n / res_spacy['elapsed_sec'] if res_spacy['elapsed_sec'] > 0 else 0

        summary.append({
            'size': n,
            'nltk_time_sec': res_nltk['elapsed_sec'],
            'spacy_time_sec': res_spacy['elapsed_sec'],
            'nltk_tweets_per_sec': tweets_per_sec_nltk,
            'spacy_tweets_per_sec': tweets_per_sec_spacy,
            'nltk_mem_before': res_nltk['mem_before'],
            'nltk_mem_after': res_nltk['mem_after'],
            'spacy_mem_before': res_spacy['mem_before'],
            'spacy_mem_after': res_spacy['mem_after'],
        })

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv('benchmark_summary.csv', index=False)
    print("\nBenchmarking complete. Summary saved to benchmark_summary.csv")
    return df_summary

if __name__ == '__main__':
    run_benchmarks(path='Tweets.csv', sizes=[1000, 5000, 10000])
