"""Log analysis utilities for training logs.

This module provides utilities for analyzing training logs, focusing on smooth loss measurement.
"""
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import mmap
import numpy as np
import os
import pandas as pd
import re
import tqdm
import typer
from scipy.ndimage import gaussian_filter1d
import tabulate

# Initialize Typer app
app = typer.Typer()

# Type aliases
PathLike = Union[str, Path]
DataFrame = pd.DataFrame

# Constants
DEFAULT_CHUNK_SIZE = 16384
DEFAULT_MAX_COUNT = 32768
DEFAULT_WINDOW_SIZE = 100

def read_file_reverse(filepath, chunk_size=16384):
    with open(filepath, 'rb') as f:
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        buffer = bytearray()
        position = mmapped_file.size()

        while position > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            mmapped_file.seek(position)
            chunk = mmapped_file.read(read_size)
            buffer = chunk + buffer

            while b'\n' in buffer:
                newline_pos = buffer.rfind(b'\n')
                yield buffer[newline_pos + 1:].decode(errors='ignore')
                buffer = buffer[:newline_pos]

        if buffer:
            yield buffer.decode(errors='ignore')

def extract_df(log_file_path:Path, MAX_CNT:int=32768)->pd.DataFrame:
    data = []
    pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| INFO     \| megatron.core.trainerv2:training_log:\d+ - iter\s+(\d+)/\s*\d+ \| .*?lm loss( only)?: ([\d\.E\+\-]+)')
    cnt = 0
    for line in read_file_reverse(log_file_path):
        match = pattern.search(line)
        if match:
            timestamp, iter_info, lm_loss = match.group(1), match.group(2), float(match.group(len(match.groups())))
            data.append([timestamp, iter_info, lm_loss])
            cnt += 1
        if cnt >= MAX_CNT:
            break
    columns = ['Timestamp', 'Iter', 'LM Loss']
    df = pd.DataFrame(data, columns=columns)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Iter'] = df['Iter'].astype(int)
    df = df.iloc[::-1]
    df.reset_index(drop=True, inplace=True)
    return df

def apply_smooth(df:pd.DataFrame, window_size:int=100)->pd.DataFrame:
    log_interval:int = df['Iter'].iloc[1] - df['Iter'].iloc[0]
    window_size = min(window_size, 0.01*len(df))
    sigma = window_size / log_interval
    df['Smoothed LM Loss'] = gaussian_filter1d(df['LM Loss'], sigma=sigma)
    return df

def update_loss_item(
    loss_item:Dict, log_file_path:str, 
    MAX_CNT:int=32768,
    target_iter:int=-1
)->Optional[Dict]:
    res = None
    df = extract_df(log_file_path, MAX_CNT)
    if len(df) == 0:
        print(f"Empty df: {log_file_path}")
        return res
    df_final_iter = df['Iter'].iloc[-1]
    log_interval:int = df['Iter'].iloc[-1] - df['Iter'].iloc[-2]
    if "ti" in loss_item and ( abs(df_final_iter//log_interval - loss_item['ti'] // log_interval) > 1 ):
        print(f"Skip {log_file_path}, final iter: {df_final_iter} != ti: {loss_item['ti']}")
        return res
    apply_smooth(df)

    if target_iter>0:
        df = df[df['Iter'] <= target_iter]
    if len(df) == 0:
        print(f"No target_iter:{target_iter} in extract df for {log_file_path}, try to increase MAX_CNT or check the log file")
    else:
        loss_item.update({
            "loss": df['LM Loss'].iloc[-1],
            "smooth loss": df['Smoothed LM Loss'].iloc[-1],
            "target_iter": target_iter,
            "final_iter": df_final_iter,
        })
    res = loss_item
    return res

def get_all_log_files(log_dir:str)->List[str]:
    LOG_FILE_PATTERN = re.compile(r'train_log_(\d+)\.txt')
    log_files = []
    for log_file in os.listdir(log_dir):
        match = LOG_FILE_PATTERN.match(log_file)
        if match:
            log_files.append(log_file)
    log_files = sorted(log_files, key=lambda x: int(LOG_FILE_PATTERN.search(x).group(1)))
    return log_files

def get_log_dir_path(base_dir:Path, dir_pattern:str) ->List[str]:
    dir_pattern:re.Pattern = re.compile(dir_pattern)
    final_log_dirs:List[str] = []

    def traverse_dirs(current_dir:str):
        sub_dirs = [
            d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))
        ]
        if not sub_dirs:
            log_files = get_all_log_files(current_dir)
            if len(log_files) > 0:
                final_log_dirs.append(current_dir)
            return
        for sub_dir in sub_dirs:
            if not dir_pattern.search(sub_dir):
                print(f"Skip {sub_dir} not match {dir_pattern.pattern}")
                continue
            traverse_dirs(os.path.join(current_dir, sub_dir))
    traverse_dirs(f"{base_dir}")
    return final_log_dirs

def get_log_file_path(base_dir:Path, dir_pattern:str) ->List[str]:
    final_log_dirs = get_log_dir_path(base_dir, dir_pattern)
    final_log_files:List[str] = []
    for final_log_dir in final_log_dirs:
        log_files = get_all_log_files(final_log_dir)
        assert len(log_files) > 0, f"No log files found in {final_log_dir}"
        final_log_files.append(
            os.path.join(final_log_dir, log_files[-1])
        )
    return final_log_files

def process_log_file_general(log_file_path, target_iter:int=-1, max_cnt:int=32768):
    loss_item = {
        "log_file_path": log_file_path,
    }
    loss_item = update_loss_item(loss_item, log_file_path, target_iter=target_iter, MAX_CNT=max_cnt)
    return loss_item

def post_process_df_general(df:pd.DataFrame)->pd.DataFrame:
    if 'log_file_path' in df.columns:
        df['exp_name'] = df['log_file_path'].apply(lambda x: x.split('/')[-2])
    return df

def log_analysis_general(log_file_paths:List[str], target_iter:Optional[int]=None, max_cnt:int=32768)->pd.DataFrame:
    print(f"len(log_file_paths): {len(log_file_paths)}")
    print(f"cpu_count(): {cpu_count()}")
    if target_iter is not None:
        print(f"target_iter: {target_iter}")
        with Pool(cpu_count()) as pool:
            loss_items = list(
                tqdm.tqdm(pool.imap(partial(process_log_file_general, target_iter=target_iter, max_cnt=max_cnt), log_file_paths), 
                        total=len(log_file_paths)))
    else:
        with Pool(cpu_count()) as pool:
            loss_items = list(
                tqdm.tqdm(pool.imap(process_log_file_general, log_file_paths), 
                        total=len(log_file_paths)))

    loss_items = [item for item in loss_items if item is not None]
    print(f"len(loss_items): {len(loss_items)}")
    df = pd.DataFrame(loss_items)
    df = post_process_df_general(df)
    return df

@app.command()
def quick_check(
    base_dir:Path, dir_pattern:str, target_iter:int=-1,
    max_cnt:int=32768,
    pretty:bool=False
):
    log_file_paths = get_log_file_path(base_dir, dir_pattern)
    df = log_analysis_general(log_file_paths, target_iter, max_cnt)
    df = df.rename(columns={"smooth loss": "smooth_loss"})
    show_df = df[["exp_name", "smooth_loss", "target_iter", "final_iter"]]
    show_df = show_df.sort_values(by="exp_name",ignore_index=True)
    if pretty:
        print(tabulate.tabulate(show_df, headers='keys', tablefmt='pretty'))
    else:
        print(show_df.to_csv(index=False, sep="\t"))

if __name__== "__main__":
    app()

