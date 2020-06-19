import numpy as np

'''
This scripts converts the 13 dimension chord array into 
a single integer in order to do a future embedding
'''

def chord_to_emb(vect):
    try:
        position = np.argwhere(vect == 1)[0][0]
        if vect[-1] == 1:
            position += 12
    except:
        position = 24
    return position

def list_chord_to_emb(list_chords):
    new_chords = []
    for chord in list_chords:
        new_chords.append([chord_to_emb(chord)])
    return np.array(new_chords)