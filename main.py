import numpy as np
from preprocess import get_features, load_data, feature_scaling
from awgn import augment_waveforms
from torchinfo import summary
import torch.nn as nn
import torch
from tqdm.auto import tqdm
import os
import yaml
from model import gru_lstm_transformer_finetune

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded


cfg = load_yaml('./configs/config.yaml')
pkl_name = './model/{}-{}.pkl'.format(cfg['data_name'], cfg['sub_name'])

# RAVDESS dataset emotions
# shift emotions left to be 0 indexed for PyTorch
emotions_dict = {
    '0': 'surprised',
    '1': 'neutral',
    '2': 'calm',
    '3': 'happy',
    '4': 'sad',
    '5': 'angry',
    '6': 'fearful',
    '7': 'disgust'
}

# Additional attributes from RAVDESS to play with
emotion_attributes = {
    '01': 'normal',
    '02': 'strong'
}

model = gru_lstm_transformer_finetune(num_emotions=len(emotions_dict)).to(cfg['device'])


def preprocess_and_save():
    # load data
    # init explicitly to prevent data leakage from past sessions, since load_data() appends
    waveforms, emotions, intensities, genders = [], [], [], []
    waveforms, emotions, intensities, genders = load_data(
        cfg['dataset'], emotion_attributes)

    print(f'Waveforms set: {len(waveforms)} samples')
    # we have 1440 waveforms but we need to know their length too; should be 3 sec * 48k = 144k
    print(f'Waveform signal length: {len(waveforms[0])}')
    print(f'Emotions set: {len(emotions)} sample labels')

    # create storage for train, validation, test sets and their indices
    train_set, valid_set, test_set = [], [], []
    X_train, X_valid, X_test = [], [], []
    y_train, y_valid, y_test = [], [], []

    # convert waveforms to array for processing
    waveforms = np.array(waveforms)

    # process each emotion separately to make sure we builf balanced train/valid/test sets
    for emotion_num in range(len(emotions_dict)):

        # find all indices of a single unique emotion
        emotion_indices = [index for index, emotion in enumerate(
            emotions) if emotion == emotion_num]

        # seed for reproducibility
        np.random.seed(69)
        # shuffle indicies
        emotion_indices = np.random.permutation(emotion_indices)

        # store dim (length) of the emotion list to make indices
        dim = len(emotion_indices)

        # store indices of training, validation and test sets in 80/10/10 proportion
        # train set is first 80%
        train_indices = emotion_indices[:int(0.8*dim)]
        # validation set is next 10% (between 80% and 90%)
        valid_indices = emotion_indices[int(0.8*dim):int(0.9*dim)]
        # test set is last 10% (between 90% - end/100%)
        test_indices = emotion_indices[int(0.9*dim):]

        # create train waveforms/labels sets
        X_train.append(waveforms[train_indices, :])
        y_train.append(
            np.array([emotion_num]*len(train_indices), dtype=np.int32))
        # create validation waveforms/labels sets
        X_valid.append(waveforms[valid_indices, :])
        y_valid.append(
            np.array([emotion_num]*len(valid_indices), dtype=np.int32))
        # create test waveforms/labels sets
        X_test.append(waveforms[test_indices, :])
        y_test.append(
            np.array([emotion_num]*len(test_indices), dtype=np.int32))

        # store indices for each emotion set to verify uniqueness between sets
        train_set.append(train_indices)
        valid_set.append(valid_indices)
        test_set.append(test_indices)

    # concatenate, in order, all waveforms back into one array
    X_train = np.concatenate(X_train, axis=0)
    X_valid = np.concatenate(X_valid, axis=0)
    X_test = np.concatenate(X_test, axis=0)

    # concatenate, in order, all emotions back into one array
    y_train = np.concatenate(y_train, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # combine and store indices for all emotions' train, validation, test sets to verify uniqueness of sets
    train_set = np.concatenate(train_set, axis=0)
    valid_set = np.concatenate(valid_set, axis=0)
    test_set = np.concatenate(test_set, axis=0)

    # check shape of each set
    print(f'Training waveforms:{X_train.shape}, y_train:{y_train.shape}')
    print(f'Validation waveforms:{X_valid.shape}, y_valid:{y_valid.shape}')
    print(f'Test waveforms:{X_test.shape}, y_test:{y_test.shape}')

    # make sure train, validation, test sets have no overlap/are unique
    # get all unique indices across all sets and how many times each index appears (count)
    uniques, count = np.unique(np.concatenate(
        [train_set, test_set, valid_set], axis=0), return_counts=True)

    # if each index appears just once, and we have 1440 such unique indices, then all sets are unique
    if sum(count == 1) == len(emotions):
        print(
            f'\nSets are unique: {sum(count==1)} samples out of {len(emotions)} are unique')
    else:
        print(
            f'\nSets are NOT unique: {sum(count==1)} samples out of {len(emotions)} are unique')

    # initialize w arrays
    # We extract MFCC features from waveforms and store in respective 'features' array
    features_train, features_valid, features_test = [], [], []

    print('Train waveforms:')  # get training set features
    features_train = get_features(X_train, features_train, cfg['sample_rate'])

    print('\n\nValidation waveforms:')  # get validation set features
    features_valid = get_features(X_valid, features_valid, cfg['sample_rate'])

    print('\n\nTest waveforms:')  # get test set features
    features_test = get_features(X_test, features_test, cfg['sample_rate'])

    print(f'\n\nFeatures set: {len(features_train)+len(features_test)+len(features_valid)} total, {len(features_train)} train, {len(features_valid)} validation, {len(features_test)} test samples')
    print(
        f'Features (MFC coefficient matrix) shape: {len(features_train[0])} mel frequency coefficients x {len(features_train[0][1])} time steps')

    # specify multiples of our dataset to add as augmented data
    multiples = 2

    print('Train waveforms:')  # augment waveforms of training set
    features_train, y_train = augment_waveforms(
        X_train, features_train, y_train, multiples, cfg['sample_rate'])

    print('\n\nValidation waveforms:')  # augment waveforms of validation set
    features_valid, y_valid = augment_waveforms(
        X_valid, features_valid, y_valid, multiples, cfg['sample_rate'])

    print('\n\nTest waveforms:')  # augment waveforms of test set
    features_test, y_test = augment_waveforms(
        X_test, features_test, y_test, multiples, cfg['sample_rate'])

    # Check new shape of extracted features and data:
    print(
        f'\n\nNative + Augmented Features set: {len(features_train)+len(features_test)+len(features_valid)} total, {len(features_train)} train, {len(features_valid)} validation, {len(features_test)} test samples')
    print(f'{len(y_train)} training sample labels, {len(y_valid)} validation sample labels, {len(y_test)} test sample labels')
    print(
        f'Features (MFCC matrix) shape: {len(features_train[0])} mel frequency coefficients x {len(features_train[0][1])} time steps')

    # need to make dummy input channel for CNN input feature tensor
    X_train = np.expand_dims(features_train, 1)
    X_valid = np.expand_dims(features_valid, 1)
    X_test = np.expand_dims(features_test, 1)

    # convert emotion labels from list back to numpy arrays for PyTorch to work with
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)

    # confiorm that we have tensor-ready 4D data array
    # should print (batch, channel, width, height) == (4320, 1, 128, 282) when multiples==2
    print(
        f'Shape of 4D feature array for input tensor: {X_train.shape} train, {X_valid.shape} validation, {X_test.shape} test')
    print(
        f'Shape of emotion labels: {y_train.shape} train, {y_valid.shape} validation, {y_test.shape} test')

    X_train, X_valid, X_test, y_train, y_valid, y_test = feature_scaling(
        X_train, X_valid, X_test, y_train, y_valid, y_test)

    # check shape of each set again
    print(f'X_train scaled:{X_train.shape}, y_train:{y_train.shape}')
    print(f'X_valid scaled:{X_valid.shape}, y_valid:{y_valid.shape}')
    print(f'X_test scaled:{X_test.shape}, y_test:{y_test.shape}')

    # open file in write mode and write data
    with open(cfg['npy_name'], 'wb') as f:
        np.save(f, X_train)
        np.save(f, X_valid)
        np.save(f, X_test)
        np.save(f, y_train)
        np.save(f, y_valid)
        np.save(f, y_test)

    print('[*] Features and labels saved.')


def criterion(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions, target=targets)


def make_train_step(model, criterion, optimizer):

    # define the training step of the training phase
    def train_step(X, Y):

        # forward pass
        output_logits, output_softmax = model(X)
        predictions = torch.argmax(output_softmax, dim=1)
        accuracy = torch.sum(Y == predictions)/float(len(Y))

        # compute loss on logits because nn.CrossEntropyLoss implements log softmax
        loss = criterion(output_logits, Y)

        # compute gradients for the optimizer to use
        loss.backward()

        # update network parameters based on gradient stored (by calling loss.backward())
        optimizer.step()

        # zero out gradients for next pass
        # pytorch accumulates gradients from backwards passes (convenient for RNNs)
        optimizer.zero_grad()

        return loss.item(), accuracy*100
    return train_step


def make_validate_fnc(model, criterion):
    def validate(X, Y):

        # don't want to update any network parameters on validation passes: don't need gradient
        # wrap in torch.no_grad to save memory and compute in validation phase:
        with torch.no_grad():

            # set model to validation phase i.e. turn off dropout and batchnorm layers
            model.eval()

            # get the model's predictions on the validation set
            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax, dim=1)

            # calculate the mean accuracy over the entire validation set
            accuracy = torch.sum(Y == predictions)/float(len(Y))

            # compute error from logits (nn.crossentropy implements softmax)
            loss = criterion(output_logits, Y)

        return loss.item(), accuracy*100, predictions
    return validate


def make_save_checkpoint():
    def save_checkpoint(optimizer, model, epoch, filename):
        checkpoint_dict = {
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint_dict, filename)
    return save_checkpoint


def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.flag = True

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            self.flag = False
            # print(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.flag = True

# create training loop for one complete epoch (entire training set)


def train(optimizer, model, num_epochs, X_train, Y_train, X_valid, Y_valid):
    global device
    early_stopping = EarlyStopping(patience=100)

    # get training set size to calculate # iterations and minibatch indices
    train_size = X_train.shape[0]

    # encountered bugs in google colab only, unless I explicitly defined optimizer in this cell...
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, weight_decay=1e-3, momentum=0.8)

    # instantiate the checkpoint save function
    save_checkpoint = make_save_checkpoint()

    # instantiate the training step function
    train_step = make_train_step(model, criterion, optimizer=optimizer)

    # instantiate the validation loop function
    validate = make_validate_fnc(model, criterion)

    # instantiate lists to hold scalar performance metrics to plot later
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):

        # set model to train phase
        model.train()

        # shuffle entire training set in each epoch to randomize minibatch order
        train_indices = np.random.permutation(train_size)

        # shuffle the training set for each epoch:
        X_train = X_train[train_indices, :, :, :]
        Y_train = Y_train[train_indices]

        # instantiate scalar values to keep track of progress after each epoch so we can stop training when appropriate
        epoch_acc = 0
        epoch_loss = 0
        num_iterations = int(train_size / cfg['minibatch'])

        # create a loop for each minibatch of 32 samples:
        for i in range(num_iterations):

            # we have to track and update minibatch position for the current minibatch
            # if we take a random batch position from a set, we almost certainly will skip some of the data in that set
            # track minibatch position based on iteration number:
            batch_start = i * cfg['minibatch']
            # ensure we don't go out of the bounds of our training set:
            batch_end = min(batch_start + cfg['minibatch'], train_size)
            # ensure we don't have an index error
            actual_batch_size = batch_end-batch_start

            # get training minibatch with all channnels and 2D feature dims
            X = X_train[batch_start:batch_end, :, :, :]
            # get training minibatch labels
            Y = Y_train[batch_start:batch_end]

            # instantiate training tensors
            X_tensor = torch.tensor(X, device=cfg['device']).float()
            Y_tensor = torch.tensor(Y, dtype=torch.long, device=cfg['device'])

            # Pass input tensors thru 1 training step (fwd+backwards pass)
            loss, acc = train_step(X_tensor, Y_tensor)

            # aggregate batch accuracy to measure progress of entire epoch
            epoch_acc += acc * actual_batch_size / train_size
            epoch_loss += loss * actual_batch_size / train_size

            # keep track of the iteration to see if the model's too slow
            print('\r'+f'Epoch {epoch}: iteration {i}/{num_iterations}', end='')

        # create tensors from validation set
        X_valid_tensor = torch.tensor(X_valid, device=cfg['device']).float()
        Y_valid_tensor = torch.tensor(
            Y_valid, dtype=torch.long, device=cfg['device'])

        # calculate validation metrics to keep track of progress; don't need predictions now
        valid_loss, valid_acc, _ = validate(X_valid_tensor, Y_valid_tensor)

        # accumulate scalar performance metrics at each epoch to track and plot later
        train_losses.append(epoch_loss)
        valid_losses.append(valid_loss)

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print(f"\n\n[*] Early Stop - {epoch} epochs")
            print(f'[*] Best training loss - {min(train_losses)}')
            print(f'[*] Best validation loss - {min(valid_losses)}')
            break

        if early_stopping.flag:
            # Save checkpoint of the model
            save_checkpoint(optimizer, model, epoch, pkl_name)

        # keep track of each epoch's progress
        print('\r'+f' Epoch {epoch} --- loss:{epoch_loss:.3f}, Epoch accuracy:{epoch_acc:.2f}%, Validation loss:{valid_loss:.3f}, Validation accuracy:{valid_acc:.2f}%', end='')


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # preprocess_and_save()

    # open file in read mode and read data
    with open(cfg['npy_name'], 'rb') as f:
        X_train = np.load(f)
        X_valid = np.load(f)
        X_test = np.load(f)
        y_train = np.load(f)
        y_valid = np.load(f)
        y_test = np.load(f)

    print('='*60)
    print('model: {} \n'.format(cfg['sub_name']))

    # Check that we've recovered the right data
    print(f'X_train:{X_train.shape}, y_train:{y_train.shape}')
    print(f'X_valid:{X_valid.shape}, y_valid:{y_valid.shape}')
    print(f'X_test:{X_test.shape}, y_test:{y_test.shape} \n')

    print(f'Device:{device}', device)  # 출력결과: cuda
    print(f'Count of using GPUs:{torch.cuda.device_count()}')
    print(f'Current cuda device:{torch.cuda.current_device()}\n')

    print('Number of trainable params: ', sum(p.numel()
          for p in model.parameters()))
    print('='*60, '\n')

    # print(model)
    summary(model, input_size=(cfg['minibatch'], 1,40,282))

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, weight_decay=1e-3, momentum=0.8)

    train(optimizer, model, cfg['num_epochs'],
          X_train, y_train, X_valid, y_valid)

    load_checkpoint(optimizer, model, pkl_name)

    # reinitialize validation function with model from chosen checkpoint
    validate = make_validate_fnc(model, criterion)

    # Convert 4D test feature set array to tensor and move to GPU
    X_test_tensor = torch.tensor(X_test, device=device).float()
    # Convert 4D test label set array to tensor and move to GPU
    y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

    # Get the model's performance metrics using the validation function we defined
    test_loss, test_acc, predicted_emotions = validate(
        X_test_tensor, y_test_tensor)

    print(f'[*] Test accuracy is {test_acc:.2f}%')


if __name__ == '__main__':
    # preprocess_and_save()
    main()
