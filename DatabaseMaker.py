
import pydub
import requests
import numpy as np
from os import path
from pydub import AudioSegment
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def chdir(path):
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


def to_wav(src: str) -> None:
    dst = src[:-3] + "wav"
    with chdir('audio_data'):
        try:
            if 'mp3' in src:
                sound = AudioSegment.from_mp3(src)
            elif 'm4a' in src:
                sound = AudioSegment.from_file(src, format='m4a')
            else:
                raise ValueError('src must be a sound file')
            sound.export(dst, format="wav")
        except Exception:
            pass
        os.remove(src)





def converter(s: str) -> str:
    if s == b'audio/mp4':
        return '.m4a'
    elif s == b'audio/vnd.wave':
        return '.wav'
    else:
        return '.mp3'


def main():
    dir_names = [r'paridae', r'corvus', r'piciformes', r'dove', r'charadriiformes']
    for name in dir_names:
        file_name = f'{name}_data\\multimedia.csv'


        data = np.genfromtxt(
            file_name,
            dtype=str,
            delimiter=',',
            converters={1: converter},
            skip_header=1,
        )

        for counter, i in enumerate(data[:, 1:3][data[:, 0] == 'Sound']):
            if counter == 1000:
                break
            file_type = i[0]
            url = i[1]
            file_name = f'{name}_audio_file{counter}{file_type}'
            request = requests.get(url, allow_redirects=True)
            with open("audio_data/" + file_name, 'wb') as f:
                f.write(request.content)
            if '.wav' not in file_name:
                to_wav(file_name)



if __name__ == '__main__':
    main()
