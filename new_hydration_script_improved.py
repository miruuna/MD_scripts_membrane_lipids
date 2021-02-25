import MDAnalysis
from MDAnalysis import analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from MDAnalysis.analysis.distances import distance_array
from tqdm import tqdm 
from pathlib import Path
from datetime import datetime
import utils.seq_manipulation as seq_manipulation
from importlib import reload
import sys
reload(seq_manipulation)

def get_hydrating_lipid(u, peptide_name, results_directory, atom_type="P"):
    start, stop, step = None, None, 100 # frames to analyse (use all)
    lipids_within = {}
    lipids_outside = {}
    seq, seq_dict = seq_manipulation.get_aa_sequence(u, 4)
    lipids_ratio = {}
    lipids_protein = {}
    start, stop, step = u.trajectory.check_slice_indices(start, stop, step)
    dict_hydr ={}
    frame_index  = 0
    all_water = {}
    dict_within = {}
    dict_outside = {}
    dict_diff ={}
    if atom_type == "N":
        first_hydration_shell = 6.4 #This defines the first hydration shell of Nitrogen in Angstrom
    else:
        first_hydration_shell = 4.3 #This defines the first hydration shell of Phosphorus in Angstrom
    for ts in tqdm(u.trajectory[0::100]):  
        frame_index += 1
        all_water[frame_index] = {}
        dict_hydr[frame_index] = {}
        dict_within[frame_index] = {}
        dict_outside[frame_index] = {}
        dict_diff[frame_index] ={}
        hydrating_P = u.select_atoms(
            f"resname POPG and name {atom_type} and (around {first_hydration_shell} (resname SOL and name OW))", 
            updating=True)
        dict_hydr[frame_index]["all_hydrating"] = len(hydrating_P)
        dict_hydr[frame_index]["all_water"] = len(u.select_atoms(f"resname SOL and name OW", updating=True))
        for res_id, amino in seq_dict.items():
            aminp = f"{res_id}-{amino}"
            all_p_around =  \
                u.select_atoms(f"resname POPG and name {atom_type} and (around 10 (global resid {res_id}))", updating=True)
            all_p_not_around =  \
                u.select_atoms(f"resname POPG and name {atom_type} and not (around 10 (global resid {res_id}))", updating=True)
            lipid_sel_around_protein = \
                hydrating_P.select_atoms(f" around 10 (global resid {res_id})", 
                                         updating=True, periodic=True)

            lipid_sel_not_around_protein = \
                hydrating_P.select_atoms(f" not (around 10 (global resid {res_id}))", 
                                         periodic=True, updating=True)
            if len(all_p_around) == 0:
                dict_within[frame_index][f"{res_id}_{amino}"] = 0
            else:
                dict_within[frame_index][f"{res_id}_{amino}"] = len(lipid_sel_around_protein)/len(all_p_around)

            if len(all_p_not_around) == 0:
                dict_outside[frame_index][f"{res_id}_{amino}"]= 0
            else:
                dict_outside[frame_index][f"{res_id}_{amino}"] = len(lipid_sel_not_around_protein)/len(all_p_not_around)
            if dict_outside[frame_index][f"{res_id}_{amino}"] ==0 or dict_within[frame_index][f"{res_id}_{amino}"]==0:
                dict_hydr[frame_index][f"{res_id}_{amino}"] = 0
            else:
                dict_hydr[frame_index][f"{res_id}_{amino}"] = dict_within[frame_index][f"{res_id}_{amino}"]/dict_outside[frame_index][f"{res_id}_{amino}"]
    df_hydr = pd.DataFrame(dict_hydr)
    df_hydr.to_csv(f"{results_directory}/hydr_vs_not_hydr_{peptide_name}.csv")
    df_within = pd.DataFrame(dict_within)
    df_within.to_csv(f"{results_directory}/hydr_less10_{peptide_name}.csv")
    df_outside = pd.DataFrame(dict_outside)
    df_outside.to_csv(f"{results_directory}/hydr_more10_{peptide_name}.csv")
    df_diff = pd.DataFrame(dict_diff)
    df_diff.to_csv(f"{results_directory}/P_sum_{peptide_name}.csv")
    return dict_hydr, dict_within, dict_outside


    
def calc_and_write_to_file(path, membrane_type, sims_type, single=None):
    
    today = datetime.today().strftime('%Y-%m-%d')
    results_directory = f"dehydration/2021-02-10"
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    #create selection
    hydr_ratio_all = {}
    within_all = {}
    outside_all = {}
    if single == "yes":
        if os.path.isdir(path):
            peptide_name = os.path.basename(path)
            print(f"Starting calculations for single peptide -- {peptide_name}")
            peptide_path = path
            u = seq_manipulation.get_universe(peptide_path)
            hydr_ratio, dict_within, dict_outside = get_hydrating_lipid(u, peptide_name, results_directory)
            print(f"{peptide_name} --- DONE")

            hydr_ratio_all[peptide_name] = hydr_ratio
            within_all[peptide_name] = dict_within
            outside_all[peptide_name] = dict_outside
    else:
        print(f"Starting calculations for multiple peptideß")
        for directory in tqdm(os.listdir(path)):
            folder = os.path.join(path,directory)
            if os.path.isdir(folder) and "_pg" in os.path.basename(folder):
                peptide_path = folder
                peptide_name = os.path.basename(peptide_path)
                my_file = Path(f"{results_directory}/hydr_more10_{peptide_name}.csv")
                if my_file.is_file():
                    print(f"Results for {peptide_name} already exists! Skipping peptide.")
                    continue
                print(f"Starting calculations for peptide -- {peptide_name}")
                u = seq_manipulation.get_universe(peptide_path)
                hydr_ratio, dict_within, dict_outside = get_hydrating_lipid(u, peptide_name, results_directory)
                print(f"{peptide_name} --- DONE")

                hydr_ratio_all[peptide_name] = hydr_ratio
                within_all[peptide_name] = dict_within
                outside_all[peptide_name] = dict_outside
        if hydr_ratio_all:
            df_csv_ratio = pd.DataFrame(hydr_ratio_all, index=0)
            df_csv_phi.to_csv(f"{results_directory}/hydr_ratio_ALL_{membrane_type}.csv")
        if within_all:
            within = pd.concat(within_all, axis=1).sum(axis=1, level=0)
            within.to_csv(f"{results_directory}/within_less10_ALL_{membrane_type}.csv")
        if outside_all:
            outside = pd.concat(outside_all, axis=1).sum(axis=1, level=0)
            outside.to_csv(f"{results_directory}/outside_more10_ALL_{membrane_type}.csv")

            

if __name__== "__main__":
    if sys.argv[1]:
        single="yes"
    else:
        single="no"
    if len(sys.argv[1]) > 20:
        p = Path(str(sys.argv[1]))
        calc_and_write_to_file(p, "pg", "my_sims")
    else:
        p = Path(f"/scratch/groups/nms_lorenz/popg_miruna/{str(sys.argv[1])}")
    calc_and_write_to_file(p, "pg", "my_sims", single="yes")
