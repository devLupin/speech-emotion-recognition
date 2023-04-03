from pathlib import Path
from tqdm import tqdm
import csv
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')


def prepare_RAVDESS_DS(path_audios):
    """
    Generation of the dataframe with the information of the dataset. The dataframe has the following structure:
     ______________________________________________________________________________________________________________________________
    |             name            |                     path                                   |     emotion      |     actor     |
    ______________________________________________________________________________________________________________________________
    |  01-01-01-01-01-01-01.wav   |    <RAVDESS_dir>/audios_16kHz/01-01-01-01-01-01-01.wav     |     Neutral      |     1         |
    ______________________________________________________________________________________________________________________________
    ...
    :param path_audios: Path to the folder that contains all the audios in .wav format, 16kHz and single-channel(mono)
    """
    dict_emotions_ravdess = {
        0: 'Neutral',
        1: 'Calm',
        2: 'Happy',
        3: 'Sad',
        4: 'Angry',
        5: 'Fear',
        6: 'Disgust',
        7: 'Surprise'
    }
    
    wav_paths, emotions = [], []
    for path in tqdm(Path(path_audios).glob("*/*.wav")):
        name = str(path).split('/')[-1].split('.')[0]
        label = int(name.split("-")[2]) - 1  # Start emotions in 0

        try:
            wav_paths.append(path)
            emotions.append(label)
        except Exception as e:
            # print(str(path), e)
            pass
        
    return wav_paths, emotions

def save_melspectrogram(save_path, wav_paths, emotions, start_idx, end_idx):
    os.makedirs(save_path, exist_ok=True)
    
    wav_paths = wav_paths[start_idx:end_idx]
    emotions = emotions[start_idx:end_idx]
    
    f = open('melspectrogram.csv', 'a', newline='')
    write = csv.writer(f)
    
    for i in tqdm(range(len(wav_paths)), desc=f'range: {start_idx}~{end_idx}'):
        cnt = i + start_idx
        
        y, sr = librosa.load(wav_paths[i], sr=16000)
        
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, win_length=512, window='hamming', hop_length=256, n_mels=256, fmax=sr/2)
        melspectrogram = librosa.power_to_db(S, ref=np.max)
        # melspectrogram = melspectrogram[:226,39:220]
        librosa.display.specshow(melspectrogram, sr=sr)
        
        melspectrogram_path = os.path.join(save_path, str(cnt)+'.png')
        plt.axis('off')
        plt.savefig(melspectrogram_path, bbox_inches='tight', pad_inches = 0)
        
        li = [melspectrogram_path, emotions[cnt]]
        write.writerow(li)



if __name__ == '__main__':
    """
        Executing it all at once is slow, so it is executed in parts.
    """

    wav_paths, emotions = prepare_RAVDESS_DS('dataset')
    wav_paths = np.asarray(wav_paths)
    
    # print(wav_paths[100:200,].shape)
    
    start_idx = 0
    end_idx = 100
    for _ in range(15):
        save_melspectrogram('melspectrogram_images', wav_paths, emotions, start_idx, end_idx)
        
        start_idx += 100
        end_idx += 100