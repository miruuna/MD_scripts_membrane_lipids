import MDAnalysis
from MDAnalysis import analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import utils.seq_manipulation as seq_manipulation
from importlib import reload
reload(seq_manipulation)
import os
import sys
from MDAnalysis.analysis import leaflet
pep_num = 4

def relative_zpos(u, peptide_name, num_peptides, membrane_type, results_folder):
    _, seq_dict = seq_manipulation.get_aa_sequence(u, 4)
    pep_selection = {
        'pep1': u.select_atoms("resid 0-24 and name CA"),
        'pep2': u.select_atoms("resid 25-49 and name CA"),
        'pep3': u.select_atoms("resid 50-74 and name CA"),
        'pep4': u.select_atoms("resid 75-99 and name CA"),
        }
    pep_num_dict ={
        'pep1': list(range(0,25)),
        'pep2': list(range(25, 50)),
        'pep3': list(range(50, 75)),
        'pep4': list(range(75, 100)),
    }
  
    p_up1=[]
    L = leaflet.LeafletFinder(u, 'name P')
    ## use split leaflets to find mean z position
    p_up_mean = np.mean(L.groups(0).positions[:,[2]])
#     p_down_mean = np.mean(L.groups(1).positions[:,[2]])
    ##This variable should allow synergy analysisb
    
    z_pos_list = []
    for ts in tqdm(u.trajectory[0::100]):
        for res_id, res_name in seq_dict.items():
            timecount1 = u.trajectory.time
            output1 = np.mean(u.select_atoms(f"name CA and resid {res_id}").positions[:,[2]].astype(float))
            amino_num = res_name
            pep_num = [k for k, v in pep_num_dict.items() if res_id in v][0]
            p_up1 = np.mean(L.groups(0).positions[:,[2]])
            z_pos_list.append((timecount1, pep_num , res_id,  amino_num, output1, p_up1))
    df = pd.DataFrame(z_pos_list, columns=["Time (ns)", "Peptide_num", "Resid", "Residue", "CA Z position", "P"])
    df['Residue']= df['Residue'].astype('str')
    df['CA Z position']= ((df['CA Z position'].astype(float)-df['P'].astype(float))/10)
    df['Time (ns)'] = df['Time (ns)'].astype(float)/1000
    df['Time (ns)'] = df['Time (ns)'].astype(int)

    df.to_csv(f"{results_folder}/zpos_{peptide_name}_{membrane_type}.csv")
    return df

def calc_and_write_to_file(path, membrane_type, sims_type, single=None):
    results_directory = f"zpos/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)
    #create selection
    all_zpos={}
    if single == "yes":
        if os.path.isdir(path):
            peptide_name = os.path.basename(path)
            print(f"Starting calculations for single peptide -- {peptide_name}")
            peptide_path = path
            u = seq_manipulation.get_universe(peptide_path)
            pep_z_pos = relative_zpos(u, peptide_name, 4, membrane_type, results_directory)
            print(f"{peptide_name} --- DONE")
            all_zpos["peptide_name"] = pep_z_pos
    else:
        print(f"Starting calculations for multiple peptideß")
        for directory in tqdm(os.listdir(path)):
            folder = os.path.join(path,directory)
            if os.path.isdir(folder) and f"_{membrane_type}" in os.path.basename(folder):
                peptide_path = folder
                peptide_name = os.path.basename(peptide_path)
                my_file = Path(f"{results_directory}/zpos_{peptide_name}_{membrane_type}.csv")
                if my_file.is_file():
                    print(f"Results for {peptide_name} already exists! Skipping peptide.")
                    continue
                print(f"Starting calculations for peptide -- {peptide_name}")
                u = seq_manipulation.get_universe(peptide_path)
                pep_z_pos = relative_zpos(u, peptide_name, 4, membrane_type, results_directory)
                print(f"{peptide_name} --- DONE")
                all_zpos["peptide_name"] = pep_z_pos
        if all_zpos:
            df_pdf = pd.concat(all_zpos, axis=1).sum(axis=1, level=0)
            df_pdf.to_csv(f"{results_folder}/z_pos_ALL_{sims_type}_{membrane_type}.csv")
        else:
            sys.exit("No simulation folders found in the path provided!")
            
            
if __name__== "__main__":
    single = "yes"
    if not sys.argv[1]:
        sys.exit('No path or peptide provided. Exiting now')
    if not sys.argv[2]:
        sys.exit('Wrong membrane type or no membrane type provided. Options are: "pepg", "pg", "pe", "pc" Exiting now')
    sims_type = sys.argv[3] if len(sys.argv)==4 else "sims"
    if len(sys.argv[1]) > 20:
        p = Path(str(sys.argv[1]))
        single = "no"
    else:
        p = Path(f"/Volumes/miru_back/my_MDs/finished_short/{str(sys.argv[1])}_{sys.argv[2]}")
    calc_and_write_to_file(p, sys.argv[2], sims_type, single=single)
