import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
import matplotlib.ticker as ticker
from sklearn import preprocessing, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pandas.api.types import is_numeric_dtype
import umap
import os
import tqdm
from adjustText import adjust_text


def get_hbonds_dfs(folder_path, membrane_type):
    filenames={}
    h_dict = {}
    peptide_list = []
    for file in os.listdir(folder_path):
        file = os.path.join(folder_path, file)
        if os.path.isfile(file) and f"_{membrane_type}" in os.path.basename(file) and "hbonds_" in os.path.basename(file):
            peptide_path = os.path.basename(file).split("_")
            peptide_name = peptide_path[1]
            peptide_list.append(peptide_path[1]+"_"+peptide_path[2])
            if "V" in peptide_path[2]:
                peptide_name = peptide_path[1] + "_"+ peptide_path[2] 
            filenames[os.path.basename(file)] = peptide_name
            new_df = pd.read_csv(file)
            new_df['Hbond_num'] = new_df.mean(axis=1)
            new_df['Residue Number'] = new_df['Residue'].astype(str).str[:-3].astype(int)
            new_df['res_type'] = new_df['Residue'].astype(str).str[-3:]
            new_df["Peptide"] = peptide_name
            new_df = new_df[["Hbond_num"]]
            h_dict[peptide_name] = new_df.T

    df_list_concat = pd.concat(h_dict, axis=0)
    processed_df = df_list_concat.reset_index(level=1, drop=True)
    processed_df = processed_df.rename(columns={k:k+1 for k in list(processed_df.columns.values)}, errors="raise")
    return processed_df, peptide_list
