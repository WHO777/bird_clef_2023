import torch
import torchaudio


class TimeShift:

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, audio):
        if torch.rand(1) < self.prob:
            shift = torch.randint(0, audio.shape[0], (1, ))
            if torch.rand(1) < 0.5:
                shift = -shift
            audio = torch.roll(audio, shift.item(), dims=0)
        return audio


class GaussianNoise:

    def __init__(self, std=None, prob=0.5):
        if std is None:
            std = [0.0025, 0.025]
        self.dist = torch.distributions.Uniform(std[0], std[1])
        self.prob = prob

    def __call__(self, audio):
        std = self.dist.sample((1, ))
        if torch.rand(1) < self.prob:
            audio += (std**0.5) * torch.randn(*audio.shape)
        return audio


class TimeFreqMask:

    def __init__(self, time_mask=30, freq_mask=20, prob=0.5):
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask)
        self.prob = prob

    def __call__(self, spec):
        if torch.rand(1) > self.prob:
            return spec

        if spec.dim() == 2:
            spec = torch.unsqueeze(spec, 0)

        spec = self.freq_masking(spec)
        spec = self.time_masking(spec)
        spec = spec[0]

        return spec


class MixUp:

    def __init__(self, alpha=0.2, prob=0.5):
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.prob = prob

    def __call__(self, specs, labels):
        if torch.random() > self.prob:
            return specs, labels

        lam = self.beta.sample().item()

        specs = lam * specs + (1 - lam) * torch.roll(specs, 1, dims=0)
        labels = lam * specs + (1 - lam) * torch.roll(labels, 1, dims=0)

        return specs, labels


class CutMix:

    def __init__(self, alpha=0.2, prob=0.5):
        ...

    def __call__(self, *args, **kwargs):
        ...


def build_augments(configs):
    augments = []
    for config in configs:
        augment_type = eval(config.type)
        del config.type
        augment = augment_type(**config)
        augments.append(augment)
    return augments
