import os


    
def get_tpr_and_xtc(simulation_path):
    xtc_file_path = None
    if os.path.isfile(f"{simulation_path}/traj_comp_combined.xtc"):
        xtc_file_path = f"{simulation_path}/traj_comp_combined.xtc"
    elif os.path.isfile(f"{simulation_path}/traj_comp.xtc"):
        xtc_file_path = f"{simulation_path}/traj_comp.xtc"
    else:
        print(f"Warning!! No xtc found for peptide {os.path.basename(simulation_path)}")
    tpr_file_path = f"{simulation_path}/md_0_1.tpr" if os.path.isfile(f"{simulation_path}/md_0_1.tpr") else None
    return tpr_file_path, xtc_file_path