# Instrument-classification
Classification of four musical instruments, [Guitar, Drums, Piano, Violin] into 
their respective classes using spectrogram analysis.

This project uses audio data for four musical instruments, 
[Guitar, Drums, Piano, Violin] and converts them into mel spectrograms
using fast Fourier Transform. It then uses CNN to classify the resultant
spectrogram images. Only 1 second of the audio clips are used for
the spectrograms.

This approach works as the spectrograms for the four instruments
show different patterns as each instrument has different
characteristic sound data. These patterns are observable by 
CNNs.

Dataset: https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset/data
The dataset contains over 2500 files of sounds of these instruments. This dataset
is relatively clean except the fact that it does not correctly label
Violin sounds in the csv sheet provided. This is handled by the 'clean_data.ipynb'
jupyter notebook.

There are three notebooks in the repository:

'clean_data.ipynb' sorts the data into appropriate directories
based on their class labels. It also handles the missing
SOUND_VIOLIN label problem.

'audio_to_spec.ipynb' converts one second of the audio files
into mel spectrograms. It first trims the audio file by selecting
the middle 1 second of it, then uses the 'librosa' library which uses fft
to get the mel spectrogram of this trimmed file. Then it scales the image
and saves it into a suitable directory. It also shows the characteristic
patterns in audio for the instrument by showing images of audio data
and the obtained spectrograms.

'Classification.ipynb' uses a CNN model to classify the
spectrograms and display the results. It uses PyTorch to
define the model and train it. I have made a custom CNN model,
which is able to get a good training accuracy of 99.5%.
Validation accuracy is approx 96%.
