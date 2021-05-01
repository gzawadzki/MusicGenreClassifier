import librosa
import os
import csv
import numpy as np
#
# # generating a dataset
header = 'filename spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 14):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'disco jazz pop rock blues classical hiphop metal'.split()
for g in genres:
    for filename in os.listdir(f'./genres_original/{g}'):
        songname = f'./genres_original/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)

        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename}  {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(spec_rolloff)} {np.mean(zcr)}'
        for f in mfcc:
            to_append += f' {np.mean(f)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())