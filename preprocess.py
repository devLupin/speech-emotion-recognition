import numpy as np
import librosa, librosa.display
import glob, os
from sklearn.preprocessing import StandardScaler

# Mel Spectrograms are not directly used as a feature in this model
# Mel Spectrograms are used in calculating MFCCs, which are a higher-level representation of pitch transition
# MFCCs work better - left the mel spectrogram function here in case anyone wants to experiment
def feature_melspectrogram(
    waveform, 
    sample_rate,
    fft = 1024,
    winlen = 512,
    window='hamming',
    hop=256,
    mels=128,
    ):
    
    # Produce the mel spectrogram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    # Using 8khz as upper frequency bound should be enough for most speech classification tasks
    melspectrogram = librosa.feature.melspectrogram(
        y=waveform, 
        sr=sample_rate, 
        n_fft=fft, 
        win_length=winlen, 
        window=window, 
        hop_length=hop, 
        n_mels=mels, 
        fmax=sample_rate/2)
    
    # convert from power (amplitude**2) to decibels
    # necessary for network to learn - doesn't converge with raw power spectrograms 
    melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
    
    return melspectrogram

def feature_mfcc(
    waveform, 
    sample_rate,
    n_mfcc = 40,
    fft = 1024,
    winlen = 512,
    window='hamming',
    #hop=256, # increases # of time steps; was not helpful
    mels=128
    ):

    # Compute the MFCCs for all STFT frames 
    # 40 mel filterbanks (n_mfcc) = 40 coefficients
    mfc_coefficients=librosa.feature.mfcc(
        y=waveform, 
        sr=sample_rate, 
        n_mfcc=n_mfcc,
        n_fft=fft, 
        win_length=winlen, 
        window=window, 
        #hop_length=hop, 
        n_mels=mels, 
        fmax=sample_rate/2
        ) 

    return mfc_coefficients

def get_features(waveforms, features, sample_rate=48000):

    # initialize counter to track progress
    file_count = 0

    # process each waveform individually to get its MFCCs
    for waveform in waveforms:
        mfccs = feature_mfcc(waveform, sample_rate)
        features.append(mfccs)
        file_count += 1
        # print progress 
        print('\r'+f' Processed {file_count}/{len(waveforms)} waveforms',end='')
    
    # return all features from list of waveforms
    return features

def get_waveforms(file, sample_rate=48000):
    
    # load an individual sample audio file
    # read the full 3 seconds of the file, cut off the first 0.5s of silence; native sample rate = 48k
    # don't need to store the sample rate that librosa.load returns
    waveform, _ = librosa.load(file, duration=3, offset=0.5, sr=sample_rate)
    
    # make sure waveform vectors are homogenous by defining explicitly
    waveform_homo = np.zeros((int(sample_rate*3,)))
    waveform_homo[:len(waveform)] = waveform
    
    # return a single file's waveform                                      
    return waveform_homo

def load_data(data_path=None, emotion_attributes=None):
    # features and labels
    emotions = []
    # raw waveforms to augment later
    waveforms = []
    # extra labels
    intensities, genders = [],[]
    # progress counter
    file_count = 0
    for file in glob.glob(data_path):
        # get file name with labels
        file_name = os.path.basename(file)
        
        # get emotion label from the sample's file
        emotion = int(file_name.split("-")[2])

        #  move surprise to 0 for cleaner behaviour with PyTorch/0-indexing
        if emotion == 8: emotion = 0 # surprise is now at 0 index; other emotion indeces unchanged

        # can convert emotion label to emotion string if desired, but
        # training on number is better; better convert to emotion string after predictions are ready
        # emotion = emotions_dict[str(emotion)]
        
        # get other labels we might want
        intensity = emotion_attributes[file_name.split("-")[3]]
        # even actors are female, odd are male
        if (int((file_name.split("-")[6]).split(".")[0]))%2==0: 
            gender = 'female' 
        else: 
            gender = 'male'
            
        # get waveform from the sample
        waveform = get_waveforms(file)
        
        # store waveforms and labels
        waveforms.append(waveform)
        emotions.append(emotion)
        intensities.append(intensity) # store intensity in case we wish to predict
        genders.append(gender) # store gender in case we wish to predict 
        
        file_count += 1
        # keep track of data loader's progress
        print('\r'+f' Processed {file_count}/{1440} audio samples',end='')
        
    return waveforms, emotions, intensities, genders

def feature_scaling(X_train, X_valid, X_test, y_train, y_valid, y_test):
    scaler = StandardScaler()

    #### Scale the training data ####
    # store shape so we can transform it back 
    N,C,H,W = X_train.shape
    # Reshape to 1D because StandardScaler operates on a 1D array
    # tell numpy to infer shape of 1D array with '-1' argument
    X_train = np.reshape(X_train, (N,-1)) 
    X_train = scaler.fit_transform(X_train)
    # Transform back to NxCxHxW 4D tensor format
    X_train = np.reshape(X_train, (N,C,H,W))

    ##### Scale the validation set ####
    N,C,H,W = X_valid.shape
    X_valid = np.reshape(X_valid, (N,-1))
    X_valid = scaler.transform(X_valid)
    X_valid = np.reshape(X_valid, (N,C,H,W))

    #### Scale the test set ####
    N,C,H,W = X_test.shape
    X_test = np.reshape(X_test, (N,-1))
    X_test = scaler.transform(X_test)
    X_test = np.reshape(X_test, (N,C,H,W))

    return X_train, X_valid, X_test, y_train, y_valid, y_test
