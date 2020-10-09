import numpy as np
import pandas as pd
import IPython
from scipy.io import wavfile
import scipy.signal
import json
import librosa
from scipy import signal
import librosa.display
import os
import sys
import glob
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
%matplotlib inline

df3 = pd.read_csv(r'../audiosets/csvsfiles/metadatathree.csv', header = 1)
connect = list(df3[df3.label == "genre"].named)
connect
filestwo = dict()
filesthree = dict()
for i in range (len(connect)):
    pathtwo = '../audiosets/recordingthree/{}'.format(df3['named'][0])
    filestwo[connect[i]] = pathtwo
    sound = AudioSegment.from_wav(filestwo[connect[i]])
    sound = sound.set_channels(1)
    paththree = '../audiosets/audiosets-WAV/{}'.format(df3['named'][0])
    filesthree[connect[i]] = paththree
    sound.export(filesthree[connect[i]], format="wav")

audio_fpath = "../audiosets/recordingthree/"
audio_clips = os.listdir(audio_fpath)
for files in os.listdir(audio_clips):
    if files.endswith(".wav"):
        per_r, info = wavfile.read(files)
        info = info / 32768




print("No. of .wav files in audio folder = ",len(audio_clips))

def fftnoise(f):
    logf = np.array(f, dtype="complex")
    setup = (len(logf) - 1) // 2
    cycles = np.random.rand(setup) * 2 * np.pi
    cycles = np.cos(cycles) + 1j * np.sin(cycles)
    logf[1 : setup + 1] *= cycles
    logf[-1 : -1 - setup : -1] = np.conj(logf[1 : setup + 1])
    return np.fft.ifft(logf).real


def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    wavesfreq = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    logf = np.zeros(samples)
    f[np.logical_and(wavesfreq >= min_freq, wavesfreq <= max_freq)] = 1
    return fftnoise(logf)


fig, ax = plt.subplots(figsize=(20,4))
ax.plot(info)


for filesnames in os.listdir(audio_clips):
    if filesnames.endswith(".wav"):
        rts, collection = wavfile.read(files)
        collection = collection / 32768

noisy = collection[len(info) : len(info) * 2]
audio_ments = collection[: len(info)]
audio_ments = audio_ments / max(noisy)
noisy = noisy / max(noisy)
counter = 1 
clips_series = info + noisy / counter
noise_clip = noise_clip / counter
audio_ments = audio_ments / counter
fig, ax = plt.subplots(figsize=(20, 4))
ax.plot(audio_ments)
IPython.display.Audio(data=audio_ments, rate=info)
fig, ax = plt.subplots(figsize=(20,4))
ax.plot(clips_series)
IPython.display.Audio(data=clips_series, rate=info)

output = removeNoise(
    audiosnippets = clips_series,
    audio_ments=audio_ments,
    n_std_thresh=2,
    prop_decrease=0.95,
    visual=True,
)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
plt.plot(output, color="red")
ax.set_xlim((0, len(output)))
plt.show()
# play back a sample of the song
IPython.display.Audio(data=output, rate=44100)