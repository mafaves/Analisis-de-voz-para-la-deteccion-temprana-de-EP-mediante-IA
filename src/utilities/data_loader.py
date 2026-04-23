import os
import pandas as pd

# Define the patient type mapping
mapping_dict = {
	'HC': 0,
	'NFC': 0,
	'AC': 1,
	'PD': 2
}

def load_audio_data(root_directory, start_with=None, exact_name = None, patient_type_mapping=mapping_dict, audio_extensions=['.wav']):
	
	"""
	Loads audio file paths and labels for different patient categories.

	Args:
	- root_directory (str): The root directory containing patient type folders (e.g., 'HC', 'NFC', etc.).
	- patient_type_mapping (dict): Mapping from patient types to numerical labels.
	- audio_extensions (list): List of valid audio extensions (e.g., ['.wav', '.mp3']).

	Returns:
	- df (pandas DataFrame): DataFrame with columns: 'Patient', 'Label', 'File Path' and 'Audio Name'
	"""
	
	data = []  # Initialize an empty list to hold file paths and labels
	
	# Loop through each patient category (e.g., 'HC', 'NFC', 'AC', 'PD')
	for patient_type in patient_type_mapping.keys():
		category_dir = os.path.join(root_directory, patient_type)
		if os.path.isdir(category_dir) and os.listdir(category_dir):
    		# Loop over each patient subdirectory inside the category directory
			for patient in os.listdir(category_dir):
				patient_dir = os.path.join(category_dir, patient)
				
				# Check if the subdirectory follows the naming pattern '[A-Z][A-Z] [0-9][0-9][0-9]'
				if os.path.isdir(patient_dir) and len(patient.split("_")) == 3:
					# Extract patient number from folder name (assuming pattern: 'HUMV_{patient_type}_{patient_number}')
					_, patient_type, patient_number = patient.split("_")
					
					# Loop over the files in the patient directory
					for file in os.listdir(patient_dir):
						if file.endswith(tuple(audio_extensions)):
							audio_path = os.path.join(patient_dir, file)
							
							# Check if the audio file starts with, for example, 'vocal'
							if start_with and not file.lower().startswith(start_with.lower()):
								continue  # Skip if the file doesn't start with the specified prefix
							if exact_name and file != exact_name:
								continue  # Skip if the file does not exactly match the given name
						
							# Extract audio type from the file name (e.g., 'vocal' from 'vocal.wav')
							audio_type = os.path.splitext(file)[0]
							# Construct id for the audio
							feature_name = f"{patient_number}_{patient_type}" 
							
							# Store the patient information and file path
							data.append({
								'Patient': f'HUMV_{patient_type}_{patient_number}',
								'Label': patient_type_mapping[patient_type],
								'File_Path': audio_path,
								'Audio Name': feature_name
							})

	
	# Convert the data list to a pandas DataFrame
	df = pd.DataFrame(data)
	
	# Check if the DataFrame is empty
	if df.empty:
		print("No audio files found.")
	else:
		print(f"Loaded {len(df)} audio files.")
		print("Label distribution:")
		value_counts = df['Label'].value_counts()
		print(value_counts)
		
	return df