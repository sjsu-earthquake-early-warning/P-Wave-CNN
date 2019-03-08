from torch.utils.data import Dataset

def split_trainset(train_val_data, train_val_labels, ratio):
    train_ratio = ratio
    
    trainsize = int(len(train_val_data) * train_ratio)
    
    trainset = train_val_data[:trainsize]
    trainlabels = train_val_labels[:trainsize]
    
    valset = train_val_data[trainsize:]
    val_labels = train_val_labels[trainsize:]

    trainset = list(zip(trainset, trainlabels))
    valset = list(zip(valset, val_labels))

    return (trainset, valset)

class Waveform2Spectrogram(object):
    def __call__(self, waveform):
        import numpy as np
        from PIL import Image
        from scipy.signal import spectrogram

        freqs, times, Sx = spectrogram(waveform, fs=100, window='hamming',
                                       nperseg=30, noverlap=0,
                                       detrend='linear', scaling='spectrum')

        # Convert range while maintaining ratio
        min1 = np.amin(Sx)
        max1 = np.amax(Sx)
        min2 = 0
        max2 = 255
        Sx = ((Sx - min1) / (max1 - min1)) * (max2 - min2) + min2

        # PIL image
        return Image.fromarray(np.uint8(Sx))

class ResnetSpectrogramDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
          data (h5py._h1.dataset.Dataset): The .h5py dataset from SCSN
          transform (callable, optional): Applied on the dataset.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform = self.data[idx][0]
        if self.transform:
            waveform = self.transform(waveform)
        return (waveform, self.data[idx][1])
  