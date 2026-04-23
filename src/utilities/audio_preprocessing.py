import librosa
import numpy as np

def process_and_split_audio(audio_path, label, ids_patient, start_time, chunk_duration, max_duration, 
                           target_sr, remove_silence, top_db, silence_duration, 
                           overlap=0, min_chunk_length=0.7):
    """
    Loads audio file, processes it, and splits into chunks with optional overlap and duration control.
    
    Args:
    - audio_path: Path to the audio file
    - label: Label associated with audio
    - ids_patient: Patient ID
    - start_time: Start time in seconds for processing
    - chunk_duration: Duration of each chunk in seconds
    - max_duration: Maximum duration to process (None = use entire audio after start_time)
    - target_sr: Target sample rate
    - remove_silence: Whether to remove silence
    - top_db: Threshold for silence removal
    - silence_duration: Max silence duration to keep
    - overlap: Overlap between chunks in seconds (default 0)
    - min_chunk_length: Minimum chunk length ratio to keep (0.7 = 70% of chunk_duration)
    
    Returns:
    - List of tuples (audio_chunk, label, ids_patient)
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Resample if needed
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    # Normalize audio
    y = librosa.util.normalize(y)
    
    # Silence removal
    if remove_silence:
        non_silent_intervals = librosa.effects.split(y, top_db=top_db)
        processed_audio = []
        max_silence_samples = int(silence_duration * target_sr)
        
        for i, (start, end) in enumerate(non_silent_intervals):
            processed_audio.append(y[start:end])
            if i < len(non_silent_intervals) - 1:
                next_start = non_silent_intervals[i+1][0]
                silence_gap = next_start - end
                if silence_gap <= max_silence_samples:
                    processed_audio.append(y[end:next_start])
                    
        y = np.concatenate(processed_audio)
    
    # Calculate start and end samples
    start_sample = int(start_time * target_sr)
    if max_duration is None:
        # Use entire audio after start_time
        end_sample = len(y)
    else:
        # Use specified max duration
        end_sample = min(start_sample + int(max_duration * target_sr), len(y))
    
    # Extract audio window
    y_window = y[start_sample:end_sample]
    total_samples = len(y_window)
    
    # Calculate chunk parameters
    chunk_length = int(chunk_duration * target_sr)
    step_size = max(1, int((chunk_duration - overlap) * target_sr))
    
    # Generate chunks
    chunks = []
    i = 0
    while i < total_samples:
        end_index = min(i + chunk_length, total_samples)
        chunk = y_window[i:end_index]
        
        # Calculate padding requirements
        actual_duration = len(chunk) / target_sr
        if actual_duration / chunk_duration >= min_chunk_length:
            # Pad if needed
            if len(chunk) < chunk_length:
                pad_size = chunk_length - len(chunk)
                left_pad = pad_size // 2
                right_pad = pad_size - left_pad
                chunk = np.pad(chunk, (left_pad, right_pad), 'constant')
            chunks.append((chunk, label, ids_patient))
        
        # Break if we've reached the end
        if end_index == total_samples:
            break
            
        # Move to next chunk with overlap
        i += step_size
    
    return chunks


def execute_preprocess_and_split(df, start_time, chunk_duration, max_duration, target_sr, 
                                remove_silence, top_db=25, silence_duration=0.5, 
                                file_path_column='File_Path', ID_column= 'Patient', overlap=0, min_chunk_length=0.7):
    """
    Processes all audio files in a DataFrame and splits into chunks.
    
    Args:
    - df: DataFrame containing audio metadata
    - start_time: Start time in seconds
    - chunk_duration: Duration of each chunk
    - max_duration: Max duration to process (None = entire audio)
    - target_sr: Target sample rate
    - remove_silence: Whether to remove silence
    - top_db: Silence threshold
    - silence_duration: Max silence duration to keep
    - file_path_column: Column name for file paths
    - overlap: Overlap between chunks
    - min_chunk_length: Minimum chunk length ratio
    
    Returns:
    - chunks_np, labels_np, ids_patients_np: Numpy arrays of chunks/labels/ids
    """
    all_chunks = []
    
    for _, row in df.iterrows():
        file_path = row[file_path_column]
        label = row['Label']
        patient_id = row[ID_column]
        
        chunks = process_and_split_audio(
            audio_path=file_path,
            label=label,
            ids_patient=patient_id,
            start_time=start_time,
            chunk_duration=chunk_duration,
            max_duration=max_duration,
            target_sr=target_sr,
            remove_silence=remove_silence,
            top_db=top_db,
            silence_duration=silence_duration,
            overlap=overlap,
            min_chunk_length=min_chunk_length
        )
        all_chunks.extend(chunks)
    
    print(f"Total chunks generated: {len(all_chunks)}")
    
    if len(all_chunks) > 0:
        chunks, labels, ids = zip(*all_chunks)
        return (
            np.array(chunks, dtype=np.float32),
            np.array(labels),
            np.array(ids)
        )
    return np.array([]), np.array([]), np.array([])