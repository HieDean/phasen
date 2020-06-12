import torch
import torch.nn as nn
import numpy as np


class FTB(torch.nn.Module):
    """docstring for FTB"""

    def __init__(self, channels=24, f_length=257):
        super(FTB, self).__init__()
        self.channels = channels
        self.c_ftb_r = 5
        self.f_length = f_length

        self.conv_1_multiply_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.c_ftb_r, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.c_ftb_r),
            nn.ReLU()
        )
        self.conv_1D = nn.Sequential(
            nn.Conv1d(self.f_length * self.c_ftb_r, self.channels, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(self.channels),
            nn.ReLU()
        )
        self.frec_fc = nn.Linear(self.f_length, self.f_length, bias=False)
        self.conv_1_multiply_1_2 = nn.Sequential(
            nn.Conv2d(2 * self.channels, self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        )

    def forward(self, inputs):
        _, _, seg_length, _ = inputs.shape

        temp = self.conv_1_multiply_1_1(inputs)  # [B,c_ftb_r,segment_length,f]

        temp = temp.view(-1, self.f_length * self.c_ftb_r, seg_length)  # [B,c_ftb_r*f,segment_length]

        temp = self.conv_1D(temp)  # [B,c_a,segment_length]

        temp = temp.view(-1, self.channels, seg_length, 1)  # [B,c_a,segment_length,1]

        temp = temp * inputs  # [B,c_a,segment_length,1]*[B,c_a,segment_length,f]

        temp = self.frec_fc(temp)  # [B,c_a,segment_length,f]

        temp = torch.cat((temp, inputs), dim=1)  # [B,2*c_a,segment_length,f]

        outputs = self.conv_1_multiply_1_2(temp)  # [B,c_a,segment_length,f]

        return outputs


class gLN(torch.nn.Module):
    """docstring for gLN"""

    def __init__(self, c_p=12):
        super(gLN, self).__init__()
        self.beta = nn.Parameter(torch.ones([1, c_p, 1, 1]), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros([1, c_p, 1, 1]), requires_grad=True)

    def forward(self, inputs):
        temp = torch.mean(inputs, 1, keepdim=True)
        temp = torch.mean(temp, 2, keepdim=True)
        mean = torch.mean(temp, 3, keepdim=True)

        temp = torch.var(inputs, 1, keepdim=True)
        temp = torch.var(temp, 2, keepdim=True)
        var = torch.var(temp, 3, keepdim=True)

        outputs = (inputs - mean) / torch.sqrt(var + 1e-12) * self.beta + self.gamma

        return outputs


class TSB(torch.nn.Module):
    """docstring for TSB"""

    def __init__(self, c_a=24, c_p=12, f_length=257):
        super(TSB, self).__init__()
        self.c_a = c_a
        self.c_p = c_p
        self.f_length = f_length

        self.ftb_1 = FTB(self.c_a, f_length=self.f_length)
        self.amp_Conv2d = nn.Sequential(
            nn.Conv2d(self.c_a, self.c_a, kernel_size=5, stride=1, padding=[2, 2]),
            nn.BatchNorm2d(self.c_a),
            nn.ReLU(),

            nn.Conv2d(self.c_a, self.c_a, kernel_size=(25, 1), stride=1, padding=[12, 0]),
            nn.BatchNorm2d(self.c_a),
            nn.ReLU(),

            nn.Conv2d(self.c_a, self.c_a, kernel_size=5, stride=1, padding=[2, 2]),
            nn.BatchNorm2d(self.c_a),
            nn.ReLU()
        )
        self.ftb_2 = FTB(self.c_a, f_length=self.f_length)

        self.phase_Conv2d_1 = nn.Conv2d(in_channels=self.c_p, out_channels=self.c_p, kernel_size=(5, 3), stride=1,
                                        padding=[2, 1])
        self.gln_1 = gLN(self.c_p)

        self.phase_Conv2d_2 = nn.Conv2d(in_channels=self.c_p, out_channels=self.c_p, kernel_size=(25, 1), stride=1,
                                        padding=[12, 0])
        self.gln_2 = gLN(self.c_p)

        self.exchange_p2a = nn.Conv2d(self.c_p, self.c_a, kernel_size=1, stride=1, padding=0)
        self.exchange_a2p = nn.Conv2d(self.c_a, self.c_p, kernel_size=1, stride=1, padding=0)

    def forward(self, amp_inputs, phase_inputs):
        temp_a = self.ftb_1(amp_inputs)  # [B,c_a,segment_length,f]

        temp_a = self.amp_Conv2d(temp_a)  # [B,c_a,segment_length,f]

        temp_a = self.ftb_2(temp_a)  # [B,c_a,segment_length,f]

        temp_p = self.phase_Conv2d_1(phase_inputs)  # [B,c_p,segment_length,f]

        temp_p = self.gln_1(temp_p)  # [B,c_p,segment_length,f]

        temp_p = self.phase_Conv2d_2(temp_p)  # [B,c_p,segment_length,f]

        temp_p = self.gln_2(temp_p)  # [B,c_p,segment_length,f]

        amp_outputs = temp_a * torch.tanh(self.exchange_p2a(temp_p))  # [B,c_a,segment_length,f]
        phase_outputs = temp_p * torch.tanh(self.exchange_a2p(temp_a))  # [B,c_p,segment_length,f]

        return amp_outputs, phase_outputs


class Phasen(torch.nn.Module):
    """docstring for phasen"""

    def __init__(self, c_a=96, c_p=48, f_length=257, lstm_hide_size=600, learn_rate=0.0005):
        super(Phasen, self).__init__()
        self.c_r = 8
        self.f_length = f_length
        self.c_a = c_a
        self.c_p = c_p
        self.lstm_hide_size = lstm_hide_size
        self.n_fft = 512
        self.window = 'hamming'
        self.win_length = 400
        self.hop_length = 160

        # a_pre
        self.amp_pre = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=self.c_a, kernel_size=(7, 1), stride=1, padding=[3, 0]),
            # nn.BatchNorm2d(self.c_a),
            # nn.ReLU(),
            nn.Conv2d(in_channels=self.c_a, out_channels=self.c_a, kernel_size=(1, 7), stride=1, padding=[0, 3]),
            # nn.BatchNorm2d(self.c_a),
            # nn.ReLU()
        )

        # p_pre
        self.phase_pre = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=self.c_p, kernel_size=(5, 3), stride=1, padding=[2, 1]),
            # nn.BatchNorm2d(self.c_p),
            # nn.ReLU(),
            nn.Conv2d(in_channels=self.c_p, out_channels=self.c_p, kernel_size=(25, 1), stride=1, padding=[12, 0]),
            # nn.BatchNorm2d(self.c_p),
            # nn.ReLU()
        )

        self.tsb_1 = TSB(self.c_a, self.c_p, f_length=self.f_length)
        self.tsb_2 = TSB(self.c_a, self.c_p, f_length=self.f_length)
        self.tsb_3 = TSB(self.c_a, self.c_p, f_length=self.f_length)

        # a_suf
        self.amp_suf_Conv2d = nn.Conv2d(in_channels=self.c_a, out_channels=self.c_r, kernel_size=1, stride=1, padding=0)
        self.amp_suf_LSTM = nn.LSTM(input_size=self.f_length * self.c_r, hidden_size=self.lstm_hide_size,
                                    batch_first=True,
                                    bidirectional=True)
        self.amp_suf_fc = nn.Sequential(
            nn.Linear(self.lstm_hide_size * 2, 600, bias=True),
            nn.ReLU(),
            nn.Linear(600, 600, bias=True),
            nn.ReLU(),
            nn.Linear(600, self.f_length, bias=True),
            nn.Sigmoid()
        )

        # p_suf
        self.phase_suf = nn.Conv2d(in_channels=self.c_p, out_channels=2, kernel_size=1, stride=1, padding=0)

        self.mse = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=1,
                                                                    verbose=True)

    def forward(self, inputs):
        self.amp_suf_LSTM.flatten_parameters()

        _, _, seg_length, _ = inputs.shape
        # inputs.requires_grad = True
        # inputs.register_hook(print_grad)

        temp_a = self.amp_pre(inputs)  # [B,c_a,segment_length,f]
        temp_p = self.phase_pre(inputs)  # [B,c_p,segment_length,f]

        temp_a, temp_p = self.tsb_1(temp_a, temp_p)  # [B,c_a,segment_length,f] [B,c_p,segment_length,f]

        temp_a, temp_p = self.tsb_2(temp_a, temp_p)

        temp_a, temp_p = self.tsb_3(temp_a, temp_p)

        # suf_a
        temp_a = self.amp_suf_Conv2d(temp_a)  # [B,c_r,segment_length,f]

        temp_a = temp_a.view(-1, seg_length, self.f_length * self.c_r)  # [B,segment_length,f*c_r]

        temp_a, _ = self.amp_suf_LSTM(temp_a)  # [B,segment_length,lstm_hide_size*2]

        est_a = self.amp_suf_fc(temp_a).unsqueeze(1)  # [B,1,segment_length,f]

        # suf_p
        temp_p = self.phase_suf(temp_p)  # [B,2,segment_length,f]

        est_p = temp_p / (torch.sqrt(temp_p[:, 0, :, :] ** 2 + temp_p[:, 1, :, :] ** 2) + 1e-12).unsqueeze(
            1)  # [B,2,segment_length,f]

        inputs_abs = torch.sqrt(inputs[:, 0, :, :] ** 2 + inputs[:, 1, :, :] ** 2).unsqueeze(
            1)  # [B,1,segment_length,f]

        est_specs = inputs_abs * est_a * est_p  # [B,1,segment_length,f]*[B,1,segment_length,f]*[B,2,segment_length,f]

        return est_specs, est_a, est_p

    def loss_cal_test_with_cpr(self, est_specs, specs):
        est_specs_abs = torch.sqrt(est_specs[:, 0, :, :] ** 2 + est_specs[:, 1, :, :] ** 2)  # [b,seg,f]
        specs_abs = torch.sqrt(specs[:, 0, :, :] ** 2 + specs[:, 1, :, :] ** 2)  # [b,seg,f]

        est_specs_abs_cpr = est_specs_abs ** 0.3  # [b,seg,f]
        specs_abs_cpr = specs_abs ** 0.3  # [b,seg,f]

        est_specs_cpr = est_specs * (est_specs_abs_cpr / (est_specs_abs + 1e-12)).unsqueeze(1)
        specs_cpr = specs * (specs_abs_cpr / (specs_abs + 1e-12)).unsqueeze(1)

        loss_a = self.mse(est_specs_abs_cpr, specs_abs_cpr)
        loss_p = self.mse(est_specs_cpr, specs_cpr)
        loss = 0.5 * loss_a + 0.5 * loss_p

        return loss

    def loss_cal_test(self, est_specs, specs):
        est_specs_abs = torch.sqrt(est_specs[:, 0, :, :] ** 2 + est_specs[:, 1, :, :] ** 2)  # [b,seg,f]
        specs_abs = torch.sqrt(specs[:, 0, :, :] ** 2 + specs[:, 1, :, :] ** 2)  # [b,seg,f]

        est_specs_abs_cpr = est_specs_abs ** 0.3  # [b,seg,f]
        specs_abs_cpr = specs_abs ** 0.3  # [b,seg,f]

        est_real_p = torch.cos(torch.atan(est_specs[:, 0, :, :] / (1e-12 + est_specs[:, 1, :, :])))
        est_imag_p = torch.sin(torch.atan(est_specs[:, 0, :, :] / (1e-12 + est_specs[:, 1, :, :])))
        real_p = torch.cos(torch.atan(specs[:, 0, :, :] / (1e-12 + specs[:, 1, :, :])))
        imag_p = torch.sin(torch.atan(specs[:, 0, :, :] / (1e-12 + specs[:, 1, :, :])))

        est_specs_cpr = torch.stack([est_specs_abs_cpr * est_real_p, est_specs_abs_cpr * est_imag_p], dim=1)
        specs_cpr = torch.stack([specs_abs_cpr * real_p, specs_abs_cpr * imag_p], dim=1)

        loss_a = self.mse(est_specs_abs_cpr, specs_abs_cpr)
        loss_p = self.mse(est_specs_cpr, specs_cpr)
        loss = 0.5 * loss_a + 0.5 * loss_p

        return loss

    def choose_windows(self, name='hamming', N=20):
        # Rect/Hanning/Hamming
        if name == 'hamming':
            window = torch.Tensor([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
        elif name == 'hanning':
            window = torch.Tensor([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
        elif name == 'rect':
            window = torch.ones(N)
        return window

    def FFT(self, x, n_fft):
        batch_size, length, _ = x.shape
        if length < n_fft:
            x = torch.cat((x, torch.zeros((n_fft - length, 2), dtype=x.dtype)), axis=1)
        elif length == n_fft:
            pass
        elif length > n_fft:
            x = x[:, :n_fft, :]

        return torch.fft(x, signal_ndim=1)

    def STFT(self, x):
        # x:batch_size,segment_length
        with torch.no_grad():
            batch_size, segment_length = x.shape
            index_suf = -1 * np.arange(self.n_fft // 2, dtype=int) - 2
            index_pre = -1 * np.arange(self.n_fft // 2, dtype=int) - segment_length + self.n_fft / 2
            x = torch.cat((x[:, index_pre], x[:], x[:, index_suf]), dim=1)

            batch_size, segment_length = x.shape
            y = torch.empty([batch_size, segment_length, 2]).cuda()
            for idx in range(batch_size):
                y[idx] = torch.stack((x[idx], torch.zeros(segment_length).cuda()), dim=1)

            n_frames = 1 + (segment_length - self.n_fft) // self.hop_length

            start = 0
            end = start + self.n_fft

            window = self.choose_windows(self.window, N=self.win_length).cuda()
            n_pad = (self.n_fft - self.win_length) // 2
            window = torch.cat((torch.zeros(n_pad).cuda(), window, torch.zeros(n_pad).cuda()), dim=0)
            window = window.view(1, -1, 1)

            tmp = torch.empty([batch_size, self.n_fft // 2 + 1, 2, n_frames]).cuda()

            for idx in range(n_frames):
                frames = window * y[:, start:end, :]  # frame:batch_size,n_fft,2

                spec = self.FFT(frames, n_fft=self.n_fft)

                tmp[:, :, :, idx] = spec[:, :self.n_fft // 2 + 1, :]

                start += self.hop_length
                end += self.hop_length

            res = torch.empty([batch_size, 2, self.n_fft // 2 + 1, n_frames]).cuda()
            res[:, 0, :, :] = tmp[:, :, 0, :]
            res[:, 1, :, :] = tmp[:, :, 1, :]
        return res.view([batch_size, 2, n_frames, self.n_fft // 2 + 1])

    def STFT_cpu(self, x):
        # x:batch_size,segment_length
        with torch.no_grad():
            batch_size, segment_length = x.shape
            index_suf = -1 * np.arange(self.n_fft // 2, dtype=int) - 2
            index_pre = -1 * np.arange(self.n_fft // 2, dtype=int) - segment_length + self.n_fft / 2
            x = torch.cat((x[:, index_pre], x[:], x[:, index_suf]), dim=1)

            batch_size, segment_length = x.shape
            y = torch.empty([batch_size, segment_length, 2])
            for idx in range(batch_size):
                y[idx] = torch.stack((x[idx], torch.zeros(segment_length)), dim=1)

            n_frames = 1 + (segment_length - self.n_fft) // self.hop_length

            start = 0
            end = start + self.n_fft

            window = self.choose_windows(self.window, N=self.win_length)
            n_pad = (self.n_fft - self.win_length) // 2
            window = torch.cat((torch.zeros(n_pad), window, torch.zeros(n_pad)), dim=0)
            window = window.view(1, -1, 1)

            tmp = torch.empty([batch_size, self.n_fft // 2 + 1, 2, n_frames])

            for idx in range(n_frames):
                frames = window * y[:, start:end, :]  # frame:batch_size,n_fft,2

                spec = self.FFT(frames, n_fft=self.n_fft)

                tmp[:, :, :, idx] = spec[:, :self.n_fft // 2 + 1, :]

                start += self.hop_length
                end += self.hop_length

            res = torch.empty([batch_size, 2, self.n_fft // 2 + 1, n_frames])
            res[:, 0, :, :] = tmp[:, :, 0, :]
            res[:, 1, :, :] = tmp[:, :, 1, :]
        return res.view([batch_size, 2, n_frames, self.n_fft // 2 + 1])
