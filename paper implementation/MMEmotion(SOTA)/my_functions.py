from tqdm import tqdm
from pathlib import Path
import pandas as pd
import torchaudio
from transformers import Wav2Vec2Processor
from transformers import EvalPrediction
import numpy as np

target_sampling_rate = 16000
model_id = 'jonatasgrosman/wav2vec2-large-xlsr-53-english'

processor = Wav2Vec2Processor.from_pretrained(model_id, )

class preprocessing:

    @staticmethod
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
        data = []
        for path in tqdm(Path(path_audios).glob("**/*.wav")):
            name = str(path).split('/')[-1].split('.')[0]
            # Start emotions in 0
            label = dict_emotions_ravdess[int(name.split("-")[2]) - 1]
            actor = int(name.split("-")[-1])

            try:
                data.append({
                    "name": name,
                    "path": path,
                    "emotion": label,
                    "actor": actor
                })
            except Exception as e:
                # print(str(path), e)
                pass
        df = pd.DataFrame(data)
        return df


def speech_file_to_array_fn(path):
    """
    Loader of audio recordings. It loads the recordings and convert them to a specific sampling rate if required, and returns
    an array with the samples of the audio.
    :param path:[str] Path to the wav file.
    :param target_sampling_rate:[int] Global variable with the expected sampling rate of the model
    """
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def label_to_id(label):

    label_list = ['Angry', 'Calm', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

def preprocess_function(examples, input_column = "path", output_column = "emotion"):
    """
    Load the recordings with their labels.
    :param examples:[DataFrame]  with the samples of the training or test sets.
    :param input_column:[str]  Column that contain the paths to the recordings
    :param output_column:[str]  Column that contain the emotion associated to each recording
    :param target_sampling_rate:[int] Global variable with the expected sampling rate of the model
    """
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    target_list = [label_to_id(label) for label in examples[output_column]]

    result = processor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)

    return result

def compute_metrics(p: EvalPrediction):
    """
    Extract the metrics of the model from the predictions.
        -MSE for regression tasks
        -Accuracy for classification tasks
    :param p: EvalPrediction: Predictions of the model.
    """
    is_regression = False
   
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}