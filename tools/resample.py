import librosa.core as lc
import numpy as np
import os
import soundfile as sf

# clean_testset_path = '../datasets/clean_testset_wav'
# noisy_testset_path = '../../datasets/noisy_testset_wav'
# clean_trainset_path = '../../datasets/clean_trainset_wav'
# noisy_trainset_path = '../../datasets/noisy_trainset_wav'

resample_path = '../datasets/clean_testset_wav'
output_path = '../datasets/clean_testset_wav_16k'

if os.path.isdir(output_path):
    pass
else:
    os.mkdir(output_path)

file_list = os.listdir(resample_path)

for idy, filename in enumerate(file_list):
    print(idy, 'processing ' + filename)
    wave, _ = lc.load(resample_path + '/' + filename, sr=16000)

    sf.write(output_path + '/' + filename, wave, 16000)
