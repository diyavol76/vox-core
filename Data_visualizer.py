import numpy as np
import pandas as pd
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
import yaml



#audio_path=r'C:\data\cv-valid-dev\cv-valid-dev\genderdata'
#audio_path=r'C:\data\random\synthetic-speech-commands-dataset\augmented_dataset\augmented_dataset\bed'
#text_path=r'C:\data\cv-valid-dev.csv'


#sample_rate, samples = wavfile.read(os.path.join(audio_path,'sample-003633.wav'))


if __name__ == '__main__':

    #If It is local test, test is True
    test = True

    with open(r'config_data_path.yaml') as f:
        config = yaml.load(stream=f, Loader=yaml.FullLoader)

    if config is not None:
        if test:
            audio_path = config['DATA_DEV']['AUDIO_PATH']
            text_path = config['DATA_DEV']['TEXqT_PATH']

    files = os.listdir(audio_path)
    print(files)
    print(len(files))
    files[0]

    for wav_file in files:
        sample_rate, samples = wavfile.read(os.path.join(audio_path, wav_file))
        print(sample_rate, samples)

        plt.plot(samples[20480:22528])
        plt.show()
