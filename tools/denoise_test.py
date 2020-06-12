import torch
import librosa.core as lc
import numpy as np
import os
import soundfile as sf
import phasen

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

log_dir = '../model/phasen.pth'

c_a = 48
c_p = 24
lstm_hide_size = 300
win_length = 400
hop_length = 160
n_fft = 512
fs = 16000


def clip(wave, t_length=1, fs=16000):
    segment_length = t_length * fs
    wave_length = wave.shape[0]
    waves = []
    if wave_length < segment_length:
        pass
    elif wave_length == segment_length:
        waves.append(wave)
    elif wave_length > segment_length:
        num = wave_length // segment_length
        for n in range(num + 1):
            if n < num:
                waves.append(wave[n * segment_length:(n + 1) * segment_length])
            elif n == num:
                waves.append(wave[wave_length - segment_length:])
    return waves


def istft(spec):
    wav = lc.istft(spec, hop_length=hop_length, win_length=win_length, window='hamming')
    return wav


def stft(wave):
    spec = lc.stft(wave, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hamming')
    spec = np.transpose(spec)
    spec = np.stack((np.real(spec), np.imag(spec)), axis=0)
    return spec


def denoise_test_dataset_3():
    # clean_testset_path = '../datasets/clean_testset_wav'
    noisy_testset_path = '../datasets/noisy_testset_wav'

    if os.path.isdir('../datasets/eval'):
        pass
    else:
        os.mkdir('../datasets/eval')

    torch.manual_seed(20)

    model = phasen.Phasen(c_a=c_a, c_p=c_p, lstm_hide_size=lstm_hide_size).cuda()

    print('load model parameters')
    checkpoint = torch.load(log_dir)
    model.load_state_dict(checkpoint['phasen'])

    print('load and process')
    noisy_testset_list = os.listdir(noisy_testset_path)

    for idy, filename in enumerate(noisy_testset_list):
        print(idy, 'processing ' + filename)
        wave, _ = lc.load(noisy_testset_path + '/' + filename, sr=fs)
        g_wave = wave
        wave_length = wave.shape[0]
        waves = clip(wave)
        specs = []
        for wave in waves:
            spec = stft(wave)
            specs.append(np.array(spec))
        specs = torch.from_numpy(np.array(specs)).cuda()

        model.eval()
        with torch.no_grad():
            est_specs, est_a, est_p = model(specs)
            est_specs = est_specs.cpu().numpy().tolist()

            wave = None
            for id, est_spec in enumerate(est_specs):
                est_spec = np.array(est_spec)
                temp = 1j * est_spec[1, :, :]
                temp += est_spec[0, :, :]
                wave_segment = istft(np.transpose(temp.squeeze()))
                if wave is None:
                    wave = wave_segment
                elif id < len(est_specs) - 1:
                    wave = np.concatenate((wave, wave_segment), axis=0)
                elif id == len(est_specs) - 1:
                    p = wave_segment.shape[0] - (wave_length - wave.shape[0])

                    # wave = np.concatenate((wave, wave_segment[p:]), axis=0)

                    # wave = wave[:wave.shape[0] - p]
                    # wave = np.concatenate((wave, wave_segment), axis=0)

                    overlap = (wave[wave.shape[0] - p:] + wave_segment[:p]) / 2
                    wave = wave[:wave.shape[0] - p]
                    wave = np.concatenate((wave, overlap, wave_segment[p:]), axis=0)

        # wave = np.concatenate((wave, np.zeros(g_wave.shape[0] - wave.shape[0])), axis=0)
        # wave = np.concatenate((wave, g_wave[wave.shape[0]:]), axis=0)
        if wave.shape[0] != g_wave.shape[0]:
            print('length do not match!')
        sf.write('../datasets/eval/output_' + filename, wave, fs)


denoise_test_dataset_3()
