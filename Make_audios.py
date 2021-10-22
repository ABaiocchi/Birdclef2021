import torchaudio
import torch
from pathlib import Path
from tqdm import tqdm
import os
from joblib import Parallel, delayed


def out_name(filename, id):
    head, tail = os.path.split(filename)
    tail, ext = os.path.splitext(tail)
    namef = os.path.join(head, "data", tail)
    return "{name}_{uid}{ext}".format(name=namef, uid=id, ext=".ogg")



def createCsvWaveform(filename):
    data, sr = torchaudio.load(filename)
    data = data.squeeze(0)
    splits = int(len(data) / sr)
    for i in range(splits):
        datatemp = data[i*sr:(i+1)*sr]
        th = torch.std(datatemp)*7
        if (max(abs(datatemp)) < th):
            continue
        outname = out_name(filename, i)
        torchaudio.save(outname, datatemp.unsqueeze(0), sr)
    os.remove(filename)


path = "train_short_audio"

for folder in tqdm(os.listdir(path)):
    subfolder_fpath = os.path.join(path, folder)  # path of the subfolder
    sounds_path = Path(subfolder_fpath).glob("*.ogg")  # paths of the audios

    results = Parallel(n_jobs=12)(
        delayed(createCsvWaveform)(filename) for filename in sounds_path
    )