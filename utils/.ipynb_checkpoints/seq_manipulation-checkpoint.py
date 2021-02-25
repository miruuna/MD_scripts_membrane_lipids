import MDAnalysis
from MDAnalysis import analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import os

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
protonated_equiv = {'HSP': 'H'}

def get_aa_sequence(universe, pep_num):
    """
    universe: MDAnalysis Universe object
    
    pep_num: int
        How many peptides the simulation contains
    """
    protein_atoms = universe.select_atoms("protein")
    prot_residues = protein_atoms.residues
    len_peptide = int(len(prot_residues)/pep_num)
    res_names = prot_residues.resnames
    single_res_names = res_names[0:len_peptide]
    res_ids = prot_residues.residues.resids
    single_res_ids = res_ids[0:len_peptide]
    resnum = len(res_names)
    aa_seq = ''
    aa_seq_dict = {}
    new_dict = {k: f"{(k%len_peptide)+1}{v}" for k,v in dict(zip(res_ids, res_names)).items()}
    for res in res_names:
        if res in d.keys():
            aa_seq += d[res]
        elif res in protonated_equiv.keys():
            aa_seq +=  protonated_equiv[res]
        else:
            print(f"Attention! Residue Unknown { res }")
            aa_seq += "?"
    return aa_seq[0:25], new_dict


def get_res_ids(universe, pep_num):
    protein_atoms = universe.select_atoms("protein")
    prot_residues = protein_atoms.residues
    
    res_names = prot_residues.resnames
    res_ids = prot_residues.residues.resids
    print(res_names, res_ids)
    
    
    
def get_peptide_info(u, peptide_name, pep_num):
    peptide = dict()
    #### Single letter sequence #####
    peptide['sequence'], amino_dict = get_aa_sequence(u,4)
    #### Peptide name #####
    peptide['peptide_name'] = peptide_name
    #### Number of peptides #####
    peptide['pepnum'] = pep_num
    peptide['resnum'] = len(peptide['sequence'])
    peptide['restot'] = peptide['pepnum']*peptide['resnum']
    peptide['starting_resid'] = 1
    peptide['Aminos'] = [v for k,v in amino_dict.items()]
    peptide['Amino_dict'] = amino_dict
    return peptide

def get_universe(peptide_path, files=False):
    xtc_file_path = None
    peptide_name = os.path.basename(peptide_path)
    if os.path.isfile(f"{peptide_path}/traj_comp.xtc"):
        xtc_file_path = f"{peptide_path}/traj_comp.xtc"
    if os.path.isfile(f"{peptide_path}/traj_comp_combined.xtc"):
        xtc_file_path = f"{peptide_path}/traj_comp_combined.xtc"
    if not xtc_file_path:
        print(f"Warning!! No xtc found for peptide {peptide_name}")
    tpr_file_path = None
    if os.path.isfile(f"{peptide_path}/md_0_1.tpr"):
        tpr_file_path = f"{peptide_path}/md_0_1.tpr"
    else:
        print(f"Warning!! No tpr found for peptide {peptide_name}")
    u = MDAnalysis.Universe(tpr_file_path, xtc_file_path)
    if files:
        return u, xtc_file_path, tpr_file_path
    else:
        return u

def get_aa_seq_dict(path, pep_num, single="No"):
    #create selection
    a_dict={}
    if single == "yes":
        if os.path.isdir(path):
            peptide_name = os.path.basename(path)
            print(f"Starting calculations for single peptide -- {peptide_name}")
            peptide_path = path
            u = get_universe(peptide_path)
            a_dict[peptide_name], _ = get_aa_sequence(u, 4)

    else:
        print(f"Starting calculations for multiple peptideß")
        for directory in tqdm(os.listdir(path)):
            folder = os.path.join(path,directory)
            if os.path.isdir(folder) and "_pg" in os.path.basename(folder):
                peptide_path = folder
                peptide_name = os.path.basename(peptide_path)

                print(f"Starting calculations for peptide -- {peptide_name}")
                u = get_universe(peptide_path)
                a_dict[peptide_name], _ = get_aa_sequence(u, 4)
    seq_dict_df = pd.DataFrame.from_dict(a_dict,  orient="index")
    seq_dict_df = seq_dict_df.reset_index()
    seq_dict_df = seq_dict_df.rename(columns={"index":"peptide", 0:"sequence"})
    seq_dict_df.to_csv("sequence_list_my_popg.csv")
    return a_dict
    

def get_aa_per_res(path, pep_num, single="No"):
    #create selection
    a_dict={}
    if single == "yes":
        if os.path.isdir(path):
            peptide_name = os.path.basename(path)
            print(f"Starting calculations for single peptide -- {peptide_name}")
            peptide_path = path
            u = get_universe(peptide_path)
            aa_seq, _ = get_aa_sequence(u, 4)
            d = {i+1:c for i,c in enumerate(aa_seq)}
            a_dict[peptide_name] = d
            

    else:
        print(f"Starting calculations for multiple peptideß")
        for directory in tqdm(os.listdir(path)):
            folder = os.path.join(path,directory)
            if os.path.isdir(folder) and "_pg" in os.path.basename(folder):
                peptide_path = folder
                peptide_name = os.path.basename(peptide_path)

                print(f"Starting calculations for peptide -- {peptide_name}")
                u = get_universe(peptide_path)
                aa_seq, _  = get_aa_sequence(u, 4)
                d = {i+1:c for i,c in enumerate(aa_seq)}
                a_dict[peptide_name] = d
    a_dict["pkr"] =  {i+1:c for i,c in enumerate("GWGSFFRRAAHVGRHVGRAALTHYL")}
    a_dict["pleu"] =  {i+1:c for i,c in enumerate("GWGSFFKKAAHVGKHVGKAALTHYL")}
    seq_dict_df = pd.DataFrame.from_dict(a_dict,  orient="index")
    seq_dict_df = seq_dict_df.reset_index()
    seq_dict_df = seq_dict_df.rename(columns={"index":"peptide", 0:"sequence"})
    seq_dict_df.to_csv("aa_per_pep_list_my_popg.csv")
    return a_dict
    
    