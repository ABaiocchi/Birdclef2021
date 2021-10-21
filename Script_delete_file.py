import os
import shutil
import tqdm
from pathlib import Path


def up_one_dir(path):
    try:
        # from Python 3.6
        parent_dir = Path(path).parents[1]
        # for Python 3.4/3.5, use str to convert the path to string
        # parent_dir = str(Path(path).parents[1])
        shutil.move(str(path), str(parent_dir))
    except IndexError:
        # no upper directory
        pass


def remove_audio():
    path = "Dataset_audio/"
    for class_ in tqdm.tqdm(os.listdir(path)):  # cycle through the folders
        audio_fpath = os.path.join(path, class_)  # path of the subfolder
        audio_clips = Path(audio_fpath).glob("*.ogg")  # path of the file audio
        print(class_)
        for audio in audio_clips:
            os.remove(audio)
            #up_one_dir(audio)
    #os.rmdir(str(audio_fpath))


# filenames = Path("birdclef-2021/noise/").glob("*")
# with open("birdclef-2021/noise/noise.txt", "w") as outfile:
#     for fname in filenames:
#         with open(fname) as infile:
#             outfile.write(infile.read())
remove_audio()
