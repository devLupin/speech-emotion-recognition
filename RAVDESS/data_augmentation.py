import numpy as np
import librosa
from preprocess import feature_mfcc


# https://melon1024.github.io/data_aug/
# Additive White Noise
def wn_waveforms(waveform):
    wave_len = len(waveform)
    white_noise = waveform + 0.005*(np.random.randn(wave_len))
    return white_noise

# shift
# don't feel any difference
def shift_waveforms(waveform, shift):
    return np.roll(waveform, shift)

# stretching
# faster according to rate
# inappropriate for emotion recognition
def stretch_waveforms(waveform, rate):
    waveform = np.asarray(waveform)
    waveform = librosa.effects.time_stretch(waveform, rate)
    return waveform


# https://github.com/IliaZenkov/transformer-cnn-emotion-recognitionb
# Clean noise compared to white noise
def awgn_waveforms(waveform, multiples=2, bits=16, snr_min=15, snr_max=30):

    # get length of waveform (should be 3*48k = 144k)
    wave_len = len(waveform)

    # Generate normally distributed (Gaussian) noises
    # one for each waveform and multiple (i.e. wave_len*multiples noises)
    noise = np.random.normal(size=(multiples, wave_len))

    # Normalize waveform and noise
    norm_constant = 2.0**(bits-1)
    norm_wave = waveform / norm_constant
    norm_noise = noise / norm_constant

    # Compute power of waveform and power of noise
    signal_power = np.sum(norm_wave ** 2) / wave_len
    noise_power = np.sum(norm_noise ** 2, axis=1) / wave_len

    # Choose random SNR in decibels in range [15,30]
    snr = np.random.randint(snr_min, snr_max)

    # Apply whitening transformation: make the Gaussian noise into Gaussian white noise
    # Compute the covariance matrix used to whiten each noise
    # actual SNR = signal/noise (power)
    # actual noise power = 10**(-snr/10)
    covariance = np.sqrt((signal_power / noise_power) * 10 ** (- snr / 10))
    # Get covariance matrix with dim: (144000, 2) so we can transform 2 noises: dim (2, 144000)
    covariance = np.ones((wave_len, multiples)) * covariance

    # Since covariance and noise are arrays, * is the haddamard product
    # Take Haddamard product of covariance and noise to generate white noise
    multiple_augmented_waveforms = waveform + covariance.T * noise

    return multiple_augmented_waveforms

def augment_awgn_waveforms(waveforms, features, emotions, multiples, sample_rate):
    # keep track of how many waveforms we've processed so we can add correct emotion label in the same order
    emotion_count = 0
    # keep track of how many augmented samples we've added
    added_count = 0
    # convert emotion array to list for more efficient appending
    emotions = emotions.tolist()

    for waveform in waveforms:

        # Generate 2 augmented multiples of the dataset, i.e. 1440 native + 1440*2 noisy = 4320 samples total
        augmented_waveforms = awgn_waveforms(waveform, multiples=multiples)

        # compute spectrogram for each of 2 augmented waveforms
        for augmented_waveform in augmented_waveforms:

            # Compute MFCCs over augmented waveforms
            augmented_mfcc = feature_mfcc(
                augmented_waveform, sample_rate=sample_rate)

            # append the augmented spectrogram to the rest of the native data
            features.append(augmented_mfcc)
            emotions.append(emotions[emotion_count])

            # keep track of new augmented samples
            added_count += 1

            # check progress
            print(
                '\r'+f'Processed {emotion_count + 1}/{len(waveforms)} waveforms for {added_count}/{len(waveforms)*multiples} new augmented samples', end='')

        # keep track of the emotion labels to append in order
        emotion_count += 1

    return features, emotions


def augment_noise_waveforms(waveforms, features, emotions, multiples, sample_rate):
    emotion_count = 0
    added_count = 0
    emotions = emotions.tolist()

    for waveform in waveforms:

        all_mfccs = []

        augmented_waveforms = awgn_waveforms(waveform, multiples=multiples)
        # compute spectrogram for each of 2 augmented waveforms
        for augmented_waveform in augmented_waveforms:
            augmented_mfcc = feature_mfcc(augmented_waveform, sample_rate=sample_rate)
            all_mfccs.append(augmented_mfcc)
            
        white_noise_waveforms = wn_waveforms(waveform)
        white_noise_mfcc = feature_mfcc(white_noise_waveforms, sample_rate=sample_rate)
        all_mfccs.append(white_noise_mfcc)

        shift = shift_waveforms(waveform, 40000)
        shift_mfcc = feature_mfcc(shift, sample_rate=sample_rate)
        all_mfccs.append(shift_mfcc)
        
        # stretch = stretch_waveforms(waveform, 1.5)
        # stretch_mfcc = feature_mfcc(stretch, sample_rate=sample_rate)
        # all_mfccs.append(stretch_mfcc)
        
        for mfcc in all_mfccs:
            features.append(mfcc)
            emotions.append(emotions[emotion_count])
        
        
        added_count += 4
        emotion_count += 1

        print('\r'+f'Processed {emotion_count}/{len(waveforms)} waveforms for {added_count}/{len(waveforms) * 4} new augmented samples', end='')

    return features, emotions