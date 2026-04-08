import pandas as pd
import numpy as np
from tqdm import tqdm
import opensmile
import os
import data_loader_HUMV
import audio_preprocessing
import parselmouth

def load_and_preprocess_data(root_directory, start_with=None, exact_name=None, patients_to_drop=None,
                             target_sr=16000, chunk_duration=5, max_duration=None, remove_silence=True):
    """
    Loads audio data using data_loader_HUMV.load_audio_data, filters patients, preprocesses and splits into chunks.

    Args:
        root_directory (str): Root directory containing audio files.
        start_with (str, optional): Filter files starting with this string.
        exact_name (str, optional): Filter files exactly matching this name.
        patients_to_drop (list, optional): List of patient IDs to drop.
        target_sr (int): Target sampling rate.
        chunk_duration (int): Duration of each chunk in seconds.
        max_duration (int, optional): Maximum duration to process.
        remove_silence (bool): Whether to remove silence.

    Returns:
        tuple: (preprocessed_chunks_data_np, labels_chunks_np, ids_np, df_filtered)
    """
    # Load the audio data
    df = data_loader_HUMV.load_audio_data(root_directory, start_with, exact_name)

    # Drop patients if specified
    if patients_to_drop:
        df_filtered = df[~df['Patient'].isin(patients_to_drop)]
    else:
        df_filtered = df

    # Filter labels: HC (0) vs PD (2 -> 1)
    df_HC_vs_PD = df_filtered[df_filtered['Label'].isin([0, 2])]
    df_HC_vs_PD = df_HC_vs_PD.copy()
    df_HC_vs_PD['Label'] = df_HC_vs_PD['Label'].replace(2, 1)

    print(f"DataFrame shape after filtering: {df_HC_vs_PD.shape}")
    print("Label distribution:")
    print(df_HC_vs_PD['Label'].value_counts())

    # Preprocess and split
    preprocessed_chunks_data_np, labels_chunks_np, ids_np = audio_preprocessing.execute_preprocess_and_split(
        df_HC_vs_PD, start_time=0, chunk_duration=chunk_duration, max_duration=max_duration,
        target_sr=target_sr, remove_silence=remove_silence, file_path_column='File_Path'
    )

    print(f"Number of chunks: {len(preprocessed_chunks_data_np)}")
    print(f"Chunk length (samples): {len(preprocessed_chunks_data_np[0])}")
    print(f"Expected chunk length: {target_sr * chunk_duration}")

    # Get unique labels and counts
    unique_labels, counts = np.unique(labels_chunks_np, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} occurrences")

    return preprocessed_chunks_data_np, labels_chunks_np, ids_np, df_filtered

###################
# OPENSMILE FEATURES #
###################


def extract_features_opensmile(preprocessed_chunks_data_np, labels_chunks_np, ids_np,
                               feature_set=opensmile.FeatureSet.ComParE_2016,
                               feature_level=opensmile.FeatureLevel.Functionals,
                               sampling_rate=16000, save_df=False, output_path=None):
    """
    Extracts features from preprocessed audio chunks using OpenSMILE.

    Args:
        preprocessed_chunks_data_np (list): List of preprocessed audio chunks.
        labels_chunks_np (np.array): Labels for the chunks.
        ids_np (np.array): IDs for the chunks.
        feature_set: OpenSMILE feature set.
        feature_level: OpenSMILE feature level.
        sampling_rate (int): Sampling rate of the audio.
        save_df (bool): Whether to save the features DataFrame.
        output_path (str, optional): Path to save the DataFrame if save_df is True.

    Returns:
        pd.DataFrame: DataFrame with extracted features.
    """
    smile = opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level,
    )

    # Initialize a list to store features for each chunk
    all_chunk_features = []

    # Loop through each chunk and process the signal extracting features with OpenSmile
    for i, chunk in tqdm(enumerate(preprocessed_chunks_data_np), total=len(preprocessed_chunks_data_np), desc="Processing chunks"):
        # Extract features for the current chunk
        features = smile.process_signal(chunk, sampling_rate=sampling_rate)

        # Convert the features (if not already a DataFrame) and append to the list
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        all_chunk_features.append(features)

    # Combine all chunk features into a single DataFrame
    training_features_df = pd.concat(all_chunk_features, ignore_index=True)

    print(f"Features DataFrame shape: {training_features_df.shape}")

    # Check for NaN values
    nan_summary = training_features_df.isna().sum()
    if len(nan_summary[nan_summary > 0]) > 0:
        print("Columns with missing values and the count of NaNs in each column:")
        print(nan_summary[nan_summary > 0])
    else:
        print("There are no columns with missing values")

    # Save if requested
    if save_df and output_path:
        training_features_df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")

    return training_features_df

##################
# PRAAT FEATURES #
##################

def extract_features_praat(preprocessed_chunks_data_np, labels_chunks_np, ids_np,
                           sampling_rate=16000, save_df=False, output_path=None):
    """
    Extracts acoustic features from preprocessed audio chunks using Praat (via Parselmouth).

    Features extracted: pitch (mean, min, max, std), formants (F1, F2, F3 mean), intensity (mean, min, max, std),
    jitter, shimmer, HNR (mean).

    Args:
        preprocessed_chunks_data_np (list): List of preprocessed audio chunks.
        labels_chunks_np (np.array): Labels for the chunks.
        ids_np (np.array): IDs for the chunks.
        sampling_rate (int): Sampling rate of the audio.
        save_df (bool): Whether to save the features DataFrame.
        output_path (str, optional): Path to save the DataFrame if save_df is True.

    Returns:
        pd.DataFrame: DataFrame with extracted features.
    """
    features_list = []

    for i, chunk in tqdm(enumerate(preprocessed_chunks_data_np), total=len(preprocessed_chunks_data_np), desc="Processing chunks with Praat"):
        try:
            #chunk = np.asarray(chunk, dtype=np.float64)
            # Create Sound object
            # sound = parselmouth.Sound(chunk, sampling_rate=sampling_rate)
            sound = parselmouth.Sound(chunk.astype("float64"), sampling_frequency=sampling_rate)

            # Pitch
            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced
            pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
            pitch_min = np.min(pitch_values) if len(pitch_values) > 0 else 0
            pitch_max = np.max(pitch_values) if len(pitch_values) > 0 else 0
            pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

            # Formants
            try:
                formant = parselmouth.praat.call(sound, "To Formant (burg)", 0.025, 5, 5500, 0.025, 50)

                f1_mean = parselmouth.praat.call(formant, "Get mean", 1, 0, 0, "Hertz")
                f2_mean = parselmouth.praat.call(formant, "Get mean", 2, 0, 0, "Hertz")
                f3_mean = parselmouth.praat.call(formant, "Get mean", 3, 0, 0, "Hertz")

            except:
                f1_mean, f2_mean, f3_mean = np.nan, np.nan, np.nan

            # Intensity
            intensity = sound.to_intensity()
            intensity_values = intensity.values[0]
            intensity_mean = np.mean(intensity_values)
            intensity_min = np.min(intensity_values)
            intensity_max = np.max(intensity_values)
            intensity_std = np.std(intensity_values)

            # Jitter and Shimmer (PointProcess)
            point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
            jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

            # Harmonics-to-Noise Ratio (HNR)
            hnr = sound.to_harmonicity_cc()
            hnr_values = hnr.values[0]
            hnr_mean = np.mean(hnr_values)

            # Collect features
            features = {
                'pitch_mean': pitch_mean,
                'pitch_min': pitch_min,
                'pitch_max': pitch_max,
                'pitch_std': pitch_std,
                'f1_mean': f1_mean,
                'f2_mean': f2_mean,
                'f3_mean': f3_mean,
                'intensity_mean': intensity_mean,
                'intensity_min': intensity_min,
                'intensity_max': intensity_max,
                'intensity_std': intensity_std,
                'jitter': jitter,
                'shimmer': shimmer,
                'hnr_mean': hnr_mean
            }
            features_list.append(features)

        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            # Append NaN features or skip
            features_list.append({k: np.nan for k in ['pitch_mean', 'pitch_min', 'pitch_max', 'pitch_std', 'f1_mean', 'f2_mean', 'f3_mean', 'intensity_mean', 'intensity_min', 'intensity_max', 'intensity_std', 'jitter', 'shimmer', 'hnr_mean']})

    # Create DataFrame
    features_df = pd.DataFrame(features_list)

    print(f"Praat Features DataFrame shape: {features_df.shape}")

    # Check for NaN values
    nan_summary = features_df.isna().sum()
    if len(nan_summary[nan_summary > 0]) > 0:
        print("Columns with missing values and the count of NaNs in each column:")
        print(nan_summary[nan_summary > 0])
    else:
        print("There are no columns with missing values")

    # Save if requested
    if save_df and output_path:
        features_df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")

    return features_df

