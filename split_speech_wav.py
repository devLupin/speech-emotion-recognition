import warnings
warnings.simplefilter(action='ignore')

# Load the API
from inaSpeechSegmenter import Segmenter
from pydub import AudioSegment

import os
from tqdm.auto import tqdm
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

seg = Segmenter()

def wav_segment(src):
    segmentation = seg(src)
    only_speech = []
    for elem in segmentation:
        if elem[0] == 'male' or elem[0] == 'female':
            only_speech.append([elem[1], elem[2]])
    
    return only_speech

def save_wav(only_speech, src, dest, format='wav'):
    if len(only_speech) == 0:
        new_wav = AudioSegment.from_wav(src)
        new_wav.export(dest, format=format)
        return
    
    start, end = 9876, -1
    
    for part in only_speech:
        if start > part[0]:
            start = part[0]
        if end < part[1]:
            end = part[1]
            
    start *= 1000
    end *= 1000
    
    new_wav = AudioSegment.from_wav(src)
    new_wav = new_wav[start:end]
    new_wav.export(dest, format=format)

def ravdess():
    root = 'RAVDESS'
    src = os.path.join(root, 'dataset')
    dest = os.path.join(root, 'only_speech')
    os.makedirs(dest, exist_ok=True)
    
    for id in tqdm(os.listdir(src), desc='data generate.....'):
        os.makedirs(os.path.join(dest, id), exist_ok=True)
        for wav in os.listdir(os.path.join(src, id)):
            cur_wav = os.path.join(src, id, wav)
            new_wav = os.path.join(dest, id, wav)
            only_speech = wav_segment(cur_wav)
            save_wav(only_speech, cur_wav, new_wav)
           
def emodb():
    root = 'EmoDB'
    src = os.path.join(root, 'wav')
    dest = os.path.join(root, 'only_speech')
    os.makedirs(dest, exist_ok=True)
    
    for wav in tqdm(os.listdir(src)):
        cur_wav = os.path.join(src, wav)
        new_wav = os.path.join(dest, wav)
        only_speech = wav_segment(cur_wav)
        save_wav(only_speech, cur_wav, new_wav)

def emofilm():
    root = 'EmoFilm'
    src = os.path.join(root, 'wav')
    dest = os.path.join(root, 'only_speech')
    os.makedirs(dest, exist_ok=True)
    
    for wav in tqdm(os.listdir(src)):
        cur_wav = os.path.join(src, wav)
        new_wav = os.path.join(dest, wav)
        only_speech = wav_segment(cur_wav)
        save_wav(only_speech, cur_wav, new_wav)

def savee():
    root = 'SAVEE'
    src = os.path.join(root, 'wav')
    dest = os.path.join(root, 'only_speech')
    os.makedirs(dest, exist_ok=True)
    
    for id in tqdm(os.listdir(src), desc='data generate.....'):
        os.makedirs(os.path.join(dest, id), exist_ok=True)
        for wav in os.listdir(os.path.join(src, id)):
            cur_wav = os.path.join(src, id, wav)
            new_wav = os.path.join(dest, id, wav)
            only_speech = wav_segment(cur_wav)
            save_wav(only_speech, cur_wav, new_wav)



if __name__ == '__main__':
    # ravdess()
    # emofilm()
    savee()