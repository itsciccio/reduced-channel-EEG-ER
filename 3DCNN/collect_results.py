import pandas as pd
import numpy as np
import sys
import os

dependency = sys.argv[1]

def extract_data(file, dependency):
    dir_to_file = f"{root}/{file}"
    current_df = pd.read_csv(dir_to_file)
    
    row_data = list(current_df.iloc[0])
    row_data = [round(i*100,2) for i in row_data]
    subjects = list(current_df.iloc[0].keys())
    sub_result = {}
    for sub,result in zip(subjects, row_data):
        sub_result[sub] = float(result)    
    
    dir_to_file = dir_to_file.replace('/','\\')
    dir_to_file = dir_to_file.split("\\")

    #model_name = dir_to_file[5]

    model_name = "3DCNN"
    baseline = dir_to_file[9]
    if baseline == "with":
        model_name+="-Baseline"
    
    domain = dir_to_file[6].split("_")[1].capitalize()

    channels = dir_to_file[7].replace("chan","")
    window_size = dir_to_file[8][0]
    step_size = window_size

    if channels == "32": img_size = 9
    else: img_size = 5
    hyperparameters = f"50 epochs; 10 folds; 4x4 kernel on {img_size}x{img_size} image"
    
    target = dir_to_file[11]
    if target == "arousal": target = "ARO"
    elif target == "valence": target = "VAL"

    mean = round((sum(sub_result.values()) / len(sub_result)),2)
    try: std = round((np.std(list(sub_result.values()))),2)
    except: std = 0.0

    final_dict = {"Dependency":dependency,"Model Name":model_name,"Domain":domain,"Channels":channels, "Window Size":window_size,"Step Size":step_size, "Hyperparams":hyperparameters, "Target":target,"Mean":mean,"Std":std}

    final_dict = {**final_dict, **sub_result}

    return final_dict

col_names = ["Dependency","Model Name","Domain","Channels", "Window Size","Step Size", "Hyperparams", "Target","Mean","Std"]
results_df = pd.DataFrame(columns=col_names)

cwd = str(os.getcwd())

for root, dirs, files in os.walk(cwd+"/result_entropy/32chan"):
    for file in files:
        if file == "individual_subject_results.csv":
            if dependency == "Dependent" and "independent" not in root:
                final_dict = extract_data(file, dependency)
                results_df = results_df.append(final_dict, ignore_index=True)
            elif dependency == "Independent" and "independent" in root:
                final_dict = extract_data(file, dependency)
                results_df = results_df.append(final_dict, ignore_index=True)

for root, dirs, files in os.walk(cwd+"/result_DE/32chan"):
    for file in files:
        if file == "individual_subject_results.csv":
            if dependency == "Dependent" and "independent" not in root:
                final_dict = extract_data(file, dependency)
                results_df = results_df.append(final_dict, ignore_index=True)
            elif dependency == "Independent" and "independent" in root:
                final_dict = extract_data(file, dependency)
                results_df = results_df.append(final_dict, ignore_index=True)

results_df.to_csv(f"3DCNN - {dependency} Results -  32chan.csv",index=False)
