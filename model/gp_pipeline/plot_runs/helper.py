import re
import os
import pandas as pd

def normalize_label(label: str) -> str:
    ''' Function to normalize the labels label - used for dotted lines'''
    label = re.sub(r'^(AL|random|LH)\s+', '', label)  
    label = re.sub(r'n=\d+', '', label)              
    label = re.sub(r'\s+', ' ', label).strip()      
    return label

def load_dataframes(method_paths):
    '''Function to find and load the dataframes'''
    dfs = {}
    for method, paths in method_paths.items():
        method_dfs = []
        for path in paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                method_dfs.append(df)
                print(f"[INFO] Files found: {path}")
            else:
                print(f"[WARN] File not found and skipped: {path}")
        if method_dfs:
            dfs[method] = method_dfs
        # else:
        #     print(f"[INFO] No valid data found for method: {method}")
    return dfs
