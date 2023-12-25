import torch
import torch.nn.functional as F
import torchaudio

CLASS_NAMES = [
    'mouwag1', 'sltnig1', 'helgui', 'wtbeat1', 'kerspa2', 'btweye2', 'reftin1',
    'grwpyt1', 'beasun2', 'soucit1', 'norfis1', 'egygoo', 'yertin1', 'carwoo1',
    'spfwea1', 'whhsaw1', 'malkin1', 'gnhsun1', 'rebhor1', 'refbar2',
    'luebus1', 'blnmou1', 'blksaw1', 'yebere1', 'afghor1', 'whrshr1',
    'pabspa1', 'afrthr1', 'crefra2', 'chtapa3', 'bswdov1', 'comsan', 'slcbou1',
    'whbcan1', 'wookin1', 'bubwar2', 'butapa1', 'yewgre1', 'edcsun3',
    'tafpri1', 'cabgre1', 'categr', 'brobab1', 'blfbus1', 'joygre1', 'abethr1',
    'grecor', 'lotcor1', 'blnwea1', 'macshr1', 'norpuf1', 'norbro1', 'combuz1',
    'huncis1', 'combul2', 'yenspu1', 'bawhor2', 'thrnig1', 'reccor', 'blcapa2',
    'litwea1', 'tacsun1', 'greegr', 'refwar2', 'strher', 'afpwag1', 'ccbeat1',
    'whctur2', 'wheslf1', 'eaywag1', 'crohor1', 'bkfruw1', 'refcro1',
    'gobwea1', 'afecuc1', 'brrwhe3', 'gyhkin1', 'gryapa1', 'blacra1',
    'didcuc1', 'trobou1', 'gbesta1', 'witswa1', 'ratcis1', 'cibwar1',
    'wbgbir1', 'bltbar1', 'blbpuf2', 'afrjac1', 'augbuz1', 'hipbab1',
    'kvbsun1', 'spwlap1', 'brcale1', 'rewsta1', 'brtcha1', 'whbtit5',
    'gycwar3', 'ruegls1', 'scrcha1', 'yebapa1', 'rindov', 'sccsun2', 'spepig1',
    'hamerk1', 'raybar1', 'loceag1', 'palpri1', 'soufis1', 'easmog1',
    'blaplo1', 'wbswea1', 'marsto1', 'spmthr1', 'rehblu1', 'reboxp1',
    'grccra1', 'brwwar1', 'chewea1', 'bkctch1', 'yebduc1', 'gyhspa1',
    'rehwea1', 'vilwea1', 'hadibi1', 'sobfly1', 'cohmar1', 'somgre1',
    'whbwea1', 'brcsta1', 'somtit4', 'gybfis1', 'gyhneg1', 'yesbar1',
    'gnbcam2', 'sacibi2', 'grewoo2', 'rerswa1', 'nobfly1', 'tamdov1',
    'spewea1', 'ndcsun2', 'strsee1', 'reccuc1', 'varsun2', 'brubru1',
    'rufcha2', 'klacuc1', 'fatwid1', 'hartur1', 'fislov1', 'afrgos1',
    'scthon1', 'blhher1', 'brosun1', 'subbus1', 'blakit1', 'norcro1',
    'purgre2', 'crheag1', 'abythr1', 'supsta1', 'lesmaw1', 'reisee2',
    'walsta1', 'vimwea1', 'shesta1', 'fotdro5', 'whcpri2', 'brctch1',
    'gobbun1', 'yespet1', 'whbcou1', 'wfbeat1', 'libeat1', 'vibsta2',
    'stusta1', 'nubwoo1', 'eswdov1', 'yebsto1', 'bagwea1', 'spemou2',
    'mabeat1', 'hunsun2', 'reedov1', 'broman1', 'bcbeat1', 'squher1',
    'afmdov1', 'bltori1', 'blhgon1', 'grbcam1', 'dutdov1', 'marsun2',
    'gytbar1', 'darbar1', 'fatrav1', 'gargan', 'yebgre1', 'slbgre1', 'grywrw1',
    'yebbar1', 'affeag1', 'litegr', 'amesun2', 'yeccan1', 'rostur1', 'bawman1',
    'eubeat1', 'sincis1', 'colsun2', 'laudov1', 'rebfir2', 'wlwwar', 'piecro1',
    'litswi1', 'rocmar2', 'blacuc1', 'chucis1', 'pygbat1', 'mcptit1',
    'afbfly1', 'gyhbus1', 'meypar1', 'lawgol', 'whihel1', 'barswa', 'brcwea1',
    'afrgrp1', 'whbcro2', 'sichor1', 'pitwhy', 'palfly2', 'brican1', 'yefcan',
    'gabgos2', 'dotbar1', 'bltapa1', 'wbrcha2', 'equaka1', 'hoopoe', 'chespa1',
    'afpfly1', 'moccha1', 'carcha1', 'chibat1', 'darter3', 'rbsrob1',
    'whbwhe3', 'quailf1', 'yetgre1', 'golher1', 'afdfly1', 'spfbar1',
    'abhori1', 'piekin1', 'woosan', 'yelbis1', 'lotlap1', 'lessts1', 'afpkin1',
    'gobsta5', 'blwlap1', 'afgfly1'
]
NUM_CLASSES = len(CLASS_NAMES)


class BirdCLEFDataset(torch.utils.data.Dataset):

    def __init__(self,
                 audio_paths,
                 labels=None,
                 spec_shape=None,
                 normalize=False,
                 sample_rate=32000,
                 audio_duration=10,
                 win_length=2048,
                 f_min=20,
                 f_max=16000,
                 n_mels=None,
                 n_fft=2048,
                 audio_augments=None,
                 spec_augments=None):

        if spec_shape is None:
            spec_shape = [128, 384]
        if n_mels is None:
            n_mels = spec_shape[0]

        self.audio_paths = audio_paths
        self.labels = labels

        self.spec_shape = spec_shape
        self.normalize = normalize
        self.sample_rate = sample_rate
        self.audio_len = sample_rate * audio_duration
        self.hop_length = int(self.audio_len // (self.spec_shape[1] - 1))
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.n_fft = n_fft

        self.audio_augments = audio_augments
        self.spec_augments = spec_augments

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        filepath = self.audio_paths[index]
        audio, orig_sample_rate = torchaudio.load(filepath)
        audio = self.to_mono(audio)

        if orig_sample_rate != self.sample_rate:
            resample = torchaudio.transforms.Resample(orig_sample_rate,
                                                      self.sample_rate)
            audio = resample(audio)

        if audio.shape[0] > self.audio_len:
            audio = self.crop_audio(audio)

        if audio.shape[0] < self.audio_len:
            audio = self.pad_audio(audio)

        if self.normalize:
            audio = self.normalize_audio(audio)

        if self.audio_augments is not None:
            for augment in self.audio_augments:
                audio = augment(audio)

        spec_height = self.spec_shape[0]
        spec_width = self.spec_shape[1]

        audio_len = audio.shape[0]
        hop_length = int(audio_len // (spec_width - 1))

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=spec_height)
        mel = mel_spectrogram(audio)

        mel = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel)

        if mel.shape[1] != spec_width:
            mel = mel[:, :spec_width]

        if self.spec_augments is not None:
            for augment in self.spec_augments:
                mel = augment(mel)

        if self.labels is not None:
            label_onehot = torch.zeros(NUM_CLASSES)
            label_onehot[self.labels[index]] = 1
            label = torch.tensor(self.labels[index])
        else:
            label, label_onehot = None, None

        image = torch.stack([mel, mel, mel])

        return {
            "image": image,
            "label": label,
            "label_onehot": label_onehot,
            'class_name': CLASS_NAMES[label]
        }

    def pad_audio(self, audio):
        pad_length = self.audio_len - audio.shape[0]
        last_dim_padding = (0, pad_length)
        audio = F.pad(audio, last_dim_padding)
        return audio

    def crop_audio(self, audio):
        return audio[:self.audio_len]

    def to_mono(self, audio):
        return torch.mean(audio, axis=0)

    def normalize_audio(self, data, min_max=True):
        mean = torch.mean(data)
        std = torch.std(data)
        data = (data - mean) / std
        if min_max:
            min = torch.min(data)
            max = torch.max(data)
            data = (data - min) / (max - min)
        return data
