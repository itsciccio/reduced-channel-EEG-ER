# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import os

args = sys.argv[:]
channel_dir = args[1]
length_dir = args[2]
with_or_not = args[3]
kernel_size = args[4]
arousal_or_valence = args[5]
dependency = args[6]
feature_type = args[7] #entropy, energy, DE

custom = False
if len(sys.argv) == 9:		
    custom = True

if custom and int(channel_dir) == 32:
    raise Exception("Custom dataset is only available for 5 channels.")

if custom: custom_dir = "_custom"
else: custom_dir = ""

cwd = str(os.getcwd())
if dependency == "independent": dir_path = cwd+f"/result_{feature_type}{custom_dir}/{channel_dir}chan/{length_dir}sec/{with_or_not}/{kernel_size}x{kernel_size}/{arousal_or_valence}/independent/"
else: dir_path = cwd+f"/result_{feature_type}{custom_dir}/{channel_dir}chan/{length_dir}sec/{with_or_not}/{kernel_size}x{kernel_size}/{arousal_or_valence}/"
print("DIR:")
print(dir_path)
model_name = "CNN"

dict_to_write = {}
dict_per_subject = {}

acc = []
i = 0
#print(dir_path)
for file in os.listdir(dir_path):
    if ".csv" in str(file):
        continue

    try:
        df = pd.read_excel(dir_path+file)
    except: 
        continue

    if i == 0:
        cols = list(df.columns)
        cols.remove('accuracy')
        for col in cols:
            dict_to_write[col] = df[col][0]
    
    sub_name = str(str(file).split("_")[0])
    try:
        dict_per_subject[sub_name] += float(df['accuracy'][0])
    except: 
        dict_per_subject[sub_name] = float(df['accuracy'][0])
    acc.append(float(df['accuracy'][0]))

dict_to_write['mean accuracy'] = np.mean(acc)
result_df = pd.DataFrame(dict_to_write, index=[0])
result_df.to_csv(dir_path+"final_result.csv", index=False)

for key,val in dict_per_subject.items():
    dict_per_subject[key] /= 10
result_df = pd.DataFrame(dict_per_subject, index=[0])
result_df.to_csv(dir_path+"individual_subject_results.csv", index=False)

#python count_accuracy_pandas.py 5 3 with 4 arousal dep