import numpy as np

OVERLAP_FACTOR = 0.5
SAMPLE_RATE = 44100
FACTOR = 20.


def short_time_fourier_transform(signal, frame_size, overlap_factor=OVERLAP_FACTOR):
    hop_size = int(frame_size - np.floor(overlap_factor * frame_size))
    samples = np.append(np.zeros(int(np.floor(frame_size / 2))), signal)
    colons = np.ceil((len(samples) - frame_size) / float(hop_size)) + 1
    samples = np.append(samples, np.zeros(frame_size))
    frames = np.lib.stride_tricks.as_strided(samples, shape=(int(colons), frame_size),
                                             strides=(samples.strides[0] * hop_size,
                                                      samples.strides[0])).copy()
    frames *= np.hanning(frame_size)
    return np.fft.rfft(frames)


def logscale_spectrogram(spectrogram, sample_rate=SAMPLE_RATE, factor=FACTOR):
    time_bins, frequency_bins = np.shape(spectrogram)

    scale = np.linspace((0, 1, frequency_bins)) ** factor
    scale *= (frequency_bins - 1) / max(scale)

    new_spectrogram = np.complex128(np.zeros((time_bins, len(scale))))

    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            new_spectrogram[:, i] = np.sum(spectrogram[:, int(scale[i]):], axis=1)
        else:
            new_spectrogram[:, i] = np.sum(spectrogram[:, int(scale[i]):int(scale[i + 1])], axis=1)
    all_frequencies = np.abs(np.fft.fftfreq(frequency_bins * 2, 1. / sample_rate)[:frequency_bins + 1])

    frequencies = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            frequencies += [np.mean(all_frequencies[int(scale[i]):])]
        else:
            frequencies += [np.mean(all_frequencies[int(scale[i]):int(scale[i + 1])])]

    return new_spectrogram, frequencies
