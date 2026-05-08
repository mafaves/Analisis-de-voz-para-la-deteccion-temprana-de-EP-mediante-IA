import os
import numpy as np
import glob


def load_preprocessed_data(processed_folder, pattern_type='audio_segments_5s_with_1s_overlap_*.npy', exercise_names=None):
    """
    Load all pre-processed audio data from the saved numpy files.
    
    Args:
        processed_folder (str): Path to the folder containing pre-processed data
                                (e.g., '/path/to/data/processed')
        pattern_type (str): Glob pattern for audio segment files. 
                           Examples: 
                           - 'audio_segments_5s_with_1s_overlap_*.npy'
                           - 'audio_segments_10s_with_no_overlap_*.npy'
                           - 'audio_segments_*.npy'
        exercise_names (list, optional): Specific exercises to load. If None, loads all.
                                          Examples: ['aueoi', 'ka', 'pa', 'habla libre', etc.]
    
    Returns:
        dict: Dictionary with structure:
        {
            'exercise_name': {
                'audio_segments': np.ndarray,
                'labels': np.ndarray,
                'patient_ids': np.ndarray,
                'exercises': np.ndarray
            },
            ...
        }
    """
    data = {}
    
    # Find all audio_segments files
    pattern = os.path.join(processed_folder, pattern_type)
    audio_files = sorted(glob.glob(pattern))
    
    if not audio_files:
        print(f"No pre-processed files found in {processed_folder} with pattern: {pattern_type}")
        return data
    
    # Extract the prefix from pattern_type to use for building other file paths
    # e.g., 'audio_segments_5s_with_1s_overlap_*.npy' -> 'audio_segments_5s_with_1s_overlap_'
    prefix = pattern_type.split('*')[0]  # Everything before the *
    
    for audio_file in audio_files:
        # Extract exercise name from filename using the prefix
        # e.g., 'audio_segments_5s_with_1s_overlap_aueoi.npy' -> 'aueoi'
        filename = os.path.basename(audio_file)
        exercise_name = filename.replace(prefix, '').replace('.npy', '')
        
        # Filter by exercise_names if specified
        if exercise_names and exercise_name not in exercise_names:
            continue
        
        try:
            # Load all associated files for this exercise using the same prefix
            audio_segments = np.load(audio_file)
            labels = np.load(os.path.join(processed_folder, f'{prefix.replace("audio_segments_", "labels_")}{exercise_name}.npy'))
            patient_ids = np.load(os.path.join(processed_folder, f'{prefix.replace("audio_segments_", "patient_ids_")}{exercise_name}.npy'))
            exercises = np.load(os.path.join(processed_folder, f'{prefix.replace("audio_segments_", "exercises_")}{exercise_name}.npy'))
            
            data[exercise_name] = {
                'audio_segments': audio_segments,
                'labels': labels,
                'patient_ids': patient_ids,
                'exercises': exercises
            }
            
            print(f"✓ Loaded '{exercise_name}': {len(audio_segments)} samples, shape {audio_segments.shape}")
            
        except FileNotFoundError as e:
            print(f"✗ Error loading '{exercise_name}': Missing file - {e}")
    
    return data


def combine_preprocessed_data(data_dict):
    """
    Combine multiple exercise datasets into a single dataset.
    
    Args:
        data_dict (dict): Output from load_preprocessed_data()
    
    Returns:
        dict: Combined dataset with keys:
              - 'audio_segments': Combined audio array
              - 'labels': Combined labels
              - 'patient_ids': Combined patient IDs
              - 'exercises': Combined exercise names
    """
    if not data_dict:
        print("No data to combine")
        return {}
    
    combined = {
        'audio_segments': np.concatenate([d['audio_segments'] for d in data_dict.values()]),
        'labels': np.concatenate([d['labels'] for d in data_dict.values()]),
        'patient_ids': np.concatenate([d['patient_ids'] for d in data_dict.values()]),
        'exercises': np.concatenate([d['exercises'] for d in data_dict.values()])
    }
    
    print(f"\n✓ Combined all exercises: {len(combined['audio_segments'])} total samples")
    print(f"  Shape: {combined['audio_segments'].shape}")
    print(f"  Label distribution: {np.bincount(combined['labels'].astype(int))}")
    
    return combined


# Quick usage examples
if __name__ == "__main__":
    processed_folder = '/home/marcos/Documentos/GitHub/TFM_code/data/processed'
    
    # Example 1: Load with default pattern (5s with 1s overlap)
    # data = load_preprocessed_data(processed_folder)
    
    # Example 2: Load with 10s chunks, no overlap
    # data = load_preprocessed_data(processed_folder, pattern_type='audio_segments_10s_with_no_overlap_*.npy')
    
    # Example 3: Load with custom pattern
    # data = load_preprocessed_data(processed_folder, pattern_type='audio_segments_custom_*.npy')
    
    # Example 4: Load only specific exercises
    # data = load_preprocessed_data(processed_folder, 
    #                                pattern_type='audio_segments_5s_with_1s_overlap_*.npy',
    #                                exercise_names=['aueoi', 'ka', 'pa'])
    
    # Combine them into one dataset
    # combined = combine_preprocessed_data(data)
