import numpy as np
import pandas as pd
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt


audio_path=r'C:\data\cv-valid-dev\cv-valid-dev\genderdata'
#audio_path=r'C:\data\random\synthetic-speech-commands-dataset\augmented_dataset\augmented_dataset\bed'
text_path=r'C:\data\cv-valid-dev.csv'
files=os.listdir(audio_path)
print(files)
files[0]
sample_rate, samples = wavfile.read(os.path.join(audio_path,'sample-003633.wav'))
for wav_file in files:
    sample_rate, samples = wavfile.read(os.path.join(audio_path,wav_file))
    print(sample_rate,samples)

    plt.plot(samples[20480:22528])
    plt.show()


