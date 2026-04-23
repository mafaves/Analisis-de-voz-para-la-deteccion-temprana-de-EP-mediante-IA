import opensmile
import pandas as pd
from tqdm import tqdm


def extract_opensmile_features(
    audio_chunks,
    labels,
    patient_ids,
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
    sampling_rate=16000,
    verbose=True
):
    """
    Extract OpenSMILE features from audio chunks.

    Args:
        audio_chunks (np.array): Array of audio waveforms.
        labels (np.array): Labels for each chunk.
        patient_ids (np.array): Patient IDs for each chunk.
        feature_set (opensmile.FeatureSet): OpenSMILE feature set.
        feature_level (opensmile.FeatureLevel): OpenSMILE feature level.
        sampling_rate (int): Sampling rate.
        verbose (bool): Show progress bar.

    Returns:
        pd.DataFrame: DataFrame with extracted features + metadata columns.
    """
    smile = opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level,
    )

    all_features = []
    iterator = tqdm(enumerate(audio_chunks), total=len(audio_chunks)) if verbose else enumerate(audio_chunks)

    for i, chunk in iterator:
        features = smile.process_signal(chunk, sampling_rate=sampling_rate)

        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        all_features.append(features)

    df = pd.concat(all_features, ignore_index=True)

    df.insert(0, 'patient_id', patient_ids)
    df.insert(1, 'label', labels)

    nan_cols = df.isnull().sum()
    cols_with_nan = nan_cols[nan_cols > 0].index.tolist()
    if cols_with_nan:
        print(f"Columns with NaN values: {cols_with_nan}")
        df = df.drop(columns=cols_with_nan)

    print(f"OpenSMILE features shape: {df.shape}")

    return df