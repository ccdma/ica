"""
ffmpegを予めインストール
"""

from eval.product import inner_matrix
import pprint
from algorithm import fastica
import numpy as np
from eval.seed import mixed_matrix
from pydub import AudioSegment
import pathlib

VOICE_PATH = pathlib.Path("resources/voice")
FILENAMES = ["irasyai02mayu.mp3", "oyasumi_yama-rei.mp3"]

max_sample_len = 0
expected_rate = None
sounds = []
for file in FILENAMES:
    path = VOICE_PATH/file
    sound = AudioSegment.from_mp3(str(path))
    sounds.append(sound)

    rate = sound.frame_rate  # サンプリングレート(Hz)
    if expected_rate == None:
        expected_rate = rate
    elif expected_rate != rate: # すべてのサンプリングレートが等しいか確認
        raise Exception("all sample rate must be same.")
    
    sample_len = len(sound.get_array_of_samples())
    if max_sample_len < sample_len:
        max_sample_len = sample_len # サンプル数の最小値を取得
    
raw_sounds = []
for sound in sounds:
    raw_sound = sound.get_array_of_samples()
    raw_sound = list(raw_sound) + [0  for _ in range(max_sample_len-len(raw_sound))] # 足りない分をパディング
    raw_sounds.append(raw_sound)

S = np.array(raw_sounds)
A = mixed_matrix(len(raw_sounds))
X = A @ S
result = fastica.fast_ica(X, _assert=False)
Y = result.Y

pprint.pprint(inner_matrix(S))
pprint.pprint(inner_matrix(X))
pprint.pprint(inner_matrix(Y))

out = AudioSegment(data=raw_sounds[0].tobytes(),
                sample_width=2,
                frame_rate=expected_rate, channels=2)
out.export("./out.mp3", format="mp3")
pass