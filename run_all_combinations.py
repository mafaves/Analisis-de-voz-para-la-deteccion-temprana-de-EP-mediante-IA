import os
import subprocess
import pandas as pd

feature_types = [ "phonological", "prosody", "articulation", "joined"] # 
exercises = ["fluencia_categorial","lectura_texto" ,"habla_libre", "robo_galletas","aueoi", "ka", "pa", "patachaka", "pataka", "ta", "uy"] # 
# exercises = ["Monologo"]
file_prefix = "./../files/all_filtered_audio_HUMV_with_AC/features_"
file_suffix = "_all_audio_HUMV_with_AC.csv"
save_dir = "./../resultados_disvoice/all_filtered_audio_HUMV_HD_vs_PD_testing_AC_2"
label_map = "0:0,1:1"

for exercise in exercises:
    for feature in feature_types:    
        file_path = f"{file_prefix}{feature}{file_suffix}"
        print(f"\n\n Running SVM pipeline for feature={feature}, exercise={exercise} \n\n")
        subprocess.run([
            #"python", "run_svm_pipeline.py",
            "python", "run_svm_pipeline_testing_AC.py",
            #"python", "run_svm_pipeline_HD_vs_AC.py",
            "--file_path", file_path,
            "--feature_type", feature,
            "--exercise", exercise,
            "--label_map", label_map,
            "--save_dir", save_dir
        ])
    import glob

# Collect all best_summary files
all_summaries = []
for csv_file in glob.glob(f"{save_dir}/**/best_summary_*.csv", recursive=True):
    df = pd.read_csv(csv_file)
    all_summaries.append(df)

# Merge into final dataframe
if all_summaries:
    final_df = pd.concat(all_summaries, ignore_index=True)
    final_df.to_csv(os.path.join(save_dir, "final_best_summary_all.csv"), index=False)
    print(f"\n Final summary saved to {save_dir}/final_best_summary_all.csv")
else:
    print("\n No best_summary CSV files found. Check if run_svm_pipeline.py is saving them correctly.")

# Collect all best_summary files
all_summaries_AC = []
for csv_file in glob.glob(f"{save_dir}/**/best_AC_summary_*.csv", recursive=True):
    df = pd.read_csv(csv_file)
    all_summaries_AC.append(df)

# Merge into final dataframe
if all_summaries_AC:
    final_df_AC = pd.concat(all_summaries_AC, ignore_index=True)
    final_df_AC.to_csv(os.path.join(save_dir, "final_best_summary_all_AC.csv"), index=False)
    print(f"\n Final summary saved to {save_dir}/final_best_summary_all_AC.csv")
else:
    print("\n No best_summary CSV files found. Check if run_svm_pipeline.py is saving them correctly.")