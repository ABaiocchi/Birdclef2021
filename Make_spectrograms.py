import os
from pathlib import Path

import pandas as pd
import tqdm
import numpy
import skimage.io
import csv
import librosa
import librosa.display
from joblib import Parallel, delayed


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def spectrogram_image(y, sr, out, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=hop_length * 2,
        hop_length=hop_length,
        fmax=16300,
    )
    mels = numpy.log(mels + 1e-9)  # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(numpy.uint8)
    img = numpy.flip(img, axis=0)  # put low frequencies at the bottom in image
    # img = 255 - img  # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(out, img)


def out_png(filename, id):
    name, ext = os.path.splitext(filename)
    return "{name}_{uid}{ext}".format(name=name, uid=id, ext=".png")


# path = "birdclef-2021/train_short_audio"
# librosa.load("birdclef-2021/train_short_audio/acowoo/XC129244.ogg", sr=44100)
# x, sr = librosa.load("birdclef-2021/train_short_audio/acowoo/XC129244.ogg", sr=44100)
# print(x.shape)
# X = librosa.stft(x[1 : 512 * 384 * 2])
# Xdb = librosa.amplitude_to_db(abs(X))
# plt.figure(figsize=(14, 5))
# librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="log")
# plt.colorbar()
# plt.show()



def audio_to_png(folder):
    hop_length = 512 * 4  # 512 * 2  # number of samples per time-step in spectrogram
    n_mels = 32  # 128  # number of bins in spectrogram. Height of image
    time_steps = 96  # 384  # number of time-steps. Width of image
    audio_fpath = os.path.join(path, folder)
    audio_clips = Path(audio_fpath).glob("*.ogg")
    for filename in audio_clips:
        y, sr = librosa.load(filename, sr=44100)
        length_samples = time_steps * hop_length
        splits = int(len(y) / length_samples)
        # extract a fixed length window
        for i in range(splits):
            start_sample = i * length_samples  # starting at beginning
            window = y[start_sample : start_sample + length_samples]
            out = out_png(filename, i)
            # convert to PNG
            spectrogram_image(
                window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels
            )


def make_csv(path):
    data_dict = {"filepath": [], "label": []}
    for class_ in tqdm.tqdm(os.listdir(path)):
        class_path = os.path.join(path, class_, "data")
        for filename in os.listdir(class_path):
            if filename.endswith(".ogg"):
                data_dict["filepath"].append(os.path.join(class_, "data", filename))
                data_dict["label"].append(class_)
    return data_dict


def find_nocall_image(folder):
    eps = 25
    window = 10
    subfolder_fpath = os.path.join(path, folder)  # path of the subfolder
    f = open(
        os.path.join(subfolder_fpath, "noise.txt"),
        "w+",
    )
    images_path = Path(subfolder_fpath).glob("*.png")  # path of the image
    for image in images_path:
        maxim = torch.zeros(128)
        tens_image = torch.Tensor(imageio.imread(image))  # [128,385]
        for i in range(385 - 2 * window):
            diff = abs(
                torch.sub(
                    torch.sum(tens_image[:, i : i + window], dim=1) / window,
                    torch.sum(tens_image[:, i + window : i + window * 2], dim=1)
                    / window,
                )
            )
            maxim[diff > maxim] = diff[diff > maxim]
        if max(maxim) < eps:
            # print(str(image))
            f.write(str(image))
            f.write("\n")
            # os.remove(image)
    f.close()



#Produci spettrogrammi

#path = "train_short_audio/"
#results = Parallel(n_jobs=12)(
#    delayed(audio_to_png)(path) for path in tqdm.tqdm(os.listdir(path))
#)


#Trova file senza chiamate

#path = "train_short_audio"
#results = Parallel(n_jobs=-1)(
#    delayed(find_nocall_image)(fold) for fold in tqdm.tqdm(os.listdir(path))
#)


#Produci CSV classi

#path = "Dataset_audio/"
#data = make_csv(path)
#df = pd.DataFrame.from_dict(data)
#df.to_csv("classes_audio.csv", index=False, header=False)

