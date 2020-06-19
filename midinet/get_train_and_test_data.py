import numpy as np
import random
import math
import time

#load data
data = np.load('data_x_augmented.npy')
prev_data = np.load('prev_x_augmented.npy')
chord_data = np.load('chords_augmented.npy')
print('data shape: {}'.format(data.shape))

song_idx = int(data.shape[0]/8)
test_ratial = 0.1
test_song_num = round(song_idx*test_ratial)
train_song_num = data.shape[0] - test_song_num
print('total song number: {}'.format(song_idx))
print('number of test song: {}, \n,number of train song: {}'.format(test_song_num,train_song_num))

#create the song idx for test data

full = np.arange(song_idx)

test_idx= random.sample(range(0,full.shape[0]),test_song_num)
test_idx = np.asarray(test_idx)
print('total {} song idx for test: {}'.format(test_idx.shape[0],test_idx))

#create the song idx for train data
train_idx = np.delete(full,test_idx)
print('total {} song idx for train: {}'.format(train_idx.shape[0],train_idx))

    

def test_data(data,test_idx):

    #save the test data and train data separately
    X_te = []
    for i in range(test_idx.shape[0]):
        stp = (test_idx[i])*8
        edp = stp + 8
        song = data[stp:edp]
        print(song.shape)
        song = song.transpose(0, 1, 3, 2)
        X_te.append(song)
        
    X_te = np.vstack(X_te)
    return X_te


def train_data(data,train_idx):

    #save the test data and train data separately
    X_tr = []
    for i in range(train_idx.shape[0]):
        stp = (train_idx[i])*8
        edp = stp + 8
        song = data[stp:edp]
        song = song.transpose(0, 1, 3, 2)
        X_tr.append(song)

    X_tr = np.vstack(X_tr)
    return X_tr


def train_chord(data, train_idx):
    chord_tr = []
    for i in range(train_idx.shape[0]):
        stp = (train_idx[i])*8
        edp = stp + 8
        chord = data[stp:edp]
        chord_tr.append(chord)
    chord_tr = np.vstack(chord_tr)
    return chord_tr

def test_chord(data, test_idx):
    chord_te = []
    for i in range(test_idx.shape[0]):
        stp = (test_idx[i])*8
        edp = stp + 8
        chord = data[stp:edp]
        chord_te.append(chord)
    chord_te = np.vstack(chord_te)
    return chord_te

# test_data
X_te = test_data(data,test_idx)
prev_X_te = test_data(prev_data,test_idx)
chord_te = test_chord(chord_data,test_idx)
np.save('X_te_augmented.npy',X_te)
np.save('prev_X_te_augmented.npy',prev_X_te)
np.save('chord_te_augmented.npy', chord_te)

print('test song completed, X_te matrix shape: {}'.format(X_te.shape))

#train_data
X_tr = train_data(data,train_idx)
prev_X_tr = train_data(prev_data,train_idx)
chord_tr = train_chord(chord_data,train_idx)
np.save('X_tr_augmented.npy',X_tr)
np.save('prev_X_tr_augmented.npy',prev_X_tr)
np.save('chord_tr_augmented.npy',chord_tr)

print('train song completed, X_tr matrix shape: {}'.format(X_tr.shape))






