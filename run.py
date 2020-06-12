import torch
import librosa.core as lc
import numpy as np
import os
import phasen

log_dir = './model/phasen.pth'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = './datasets/'

noisy_trainset_path = path + 'noisy_trainset_wav_clip'
clean_trainset_path = path + 'clean_trainset_wav_clip'
noisy_testset_path = path + 'noisy_testset_wav_clip'
clean_testset_path = path + 'clean_testset_wav_clip'

c_a = 48
c_p = 24
lstm_hide_size = 300
win_length = 400
hop_length = 160
n_fft = 512
fs = 16000

batch_size = 12
epochs = 40
learn_rate = 0.0005


def load_wave_list():
    print('wave list loading...')

    noisy_testset_list = os.listdir(noisy_testset_path)
    clean_testset_list = os.listdir(clean_testset_path)
    noisy_trainset_list = os.listdir(noisy_trainset_path)
    clean_trainset_list = os.listdir(clean_trainset_path)

    clean_testset_list = [clean_testset_list[i:i + batch_size] for i in
                          range(0, len(clean_testset_list), batch_size)]
    noisy_testset_list = [noisy_testset_list[i:i + batch_size] for i in
                          range(0, len(noisy_testset_list), batch_size)]
    clean_trainset_list = [clean_trainset_list[i:i + batch_size] for i in
                           range(0, len(clean_trainset_list), batch_size)]
    noisy_trainset_list = [noisy_trainset_list[i:i + batch_size] for i in
                           range(0, len(noisy_trainset_list), batch_size)]

    return clean_testset_list, noisy_testset_list, clean_trainset_list, noisy_trainset_list


def load_wave(path, waves_list, sr):
    waves = []
    for filename in waves_list:
        wave, _ = lc.load(path + '/' + filename, sr=sr)
        waves.append(wave)
    return torch.from_numpy(np.array(waves)).cuda()


def main():
    torch.manual_seed(20)
    model = phasen.Phasen(c_a=c_a, c_p=c_p, lstm_hide_size=lstm_hide_size, learn_rate=learn_rate).cuda()

    clean_testset_list, noisy_testset_list, clean_trainset_list, noisy_trainset_list = load_wave_list()

    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['phasen'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('load epoch {} successfully!'.format(start_epoch))
    else:
        start_epoch = 0
        print('train start!')

    for epoch in range(start_epoch, epochs):
        model.train()
        for idx, [clean_waves_list, noisy_waves_list] in enumerate(zip(clean_trainset_list, noisy_trainset_list)):

            noisy_waves = load_wave(noisy_trainset_path, noisy_waves_list, sr=fs)
            clean_waves = load_wave(clean_trainset_path, clean_waves_list, sr=fs)

            noisy_specs = model.STFT(noisy_waves)
            clean_specs = model.STFT(clean_waves)

            est_specs, est_a, est_p = model(noisy_specs)

            loss = model.loss_cal_test_with_cpr(est_specs, clean_specs)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            model.optimizer.step()
            model.optimizer.zero_grad()

            if idx % 100 == 0 and idx > 0:
                print('epoch:', epoch, ' idx:', idx, ' train_loss:', loss)

        model.eval()
        with torch.no_grad():
            loss = 0
            for idx, [clean_waves_list, noisy_waves_list] in enumerate(zip(clean_testset_list, noisy_testset_list)):
                noisy_waves = load_wave(noisy_testset_path, noisy_waves_list, sr=fs)
                clean_waves = load_wave(clean_testset_path, clean_waves_list, sr=fs)

                noisy_specs = model.STFT(noisy_waves)
                clean_specs = model.STFT(clean_waves)

                est_specs, est_a, est_p = model(noisy_specs)

                loss_temp = model.loss_cal_test_with_cpr(est_specs, clean_specs)
                loss += loss_temp

            loss = loss / len(noisy_testset_list)

            print('epoch:', epoch, ' test_loss:', loss)

        model.scheduler.step(loss)

        state = {'phasen': model.state_dict(), 'optimizer': model.optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, log_dir)


main()
