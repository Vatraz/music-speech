# music-speech
Music-Speech is a system that classifies an audio sample into music or not music, with simple GUI that allows to 
analyze the content of an internet radio broadcast. The only feature of the current version of radio analyzer is to
inform the user about a class of audio currently broadcasting by selected internet radio station. This will be changed 
in the future.

The project was written in Python 3.6 using Keras, NumPy, Pandas, [pydub](https://github.com/jiaaro/pydub), 
[PySimpleGUI](https://github.com/PySimpleGUI)

Tested with over 20 hours of recordings of radio broadcasts, the accuracy of the classifier was 93.83%.

## Usage
Steps necessary to install the project:
1. ``git clone https://github.com/Vatraz/music-speech.git``
2. ``cd music-speech/``
3. ``virtualenv venv``
4. ``source venv/bin/activate``
5. ``pip install -r requirements.txt``

To run the radio application, start the script ``app.py`` by executing the command ``python app.py``. 

To start the process of analyzing a radio station, paste its link in the 'Link' field or select a .m3u file by 
pressing the 'Open' button. Then press 'Start'.


## Methodology
The classifier is a simple neural network implemented with Keras library. Based on five elements of the input vector,
the output takes values from 0 to 1, where 1 corresponds to music and 0 corresponds to not music.

The dataset used to train the classifier contained data from [GTZAN music/speech collection]( http://marsyas.info/) 
and 4 hours of recordings of radio stations, both of which were converted to the format accepted by the classifier.

Classification is performed based on 5 features extracted from a 2-second audio slice. The audio slice on which the
classification is performed must be a 16bps (bits per sample) WAV file sampled with frequency 22.05kHz. 

All features are based on a time-domain representation of the audio signal. For each frame (100 frames x 441 samples) 
of the 2-second slice (window) ZCR (Zero Crossing Rate) and STE (Short Time Energy) are calculated. Based on ZCR and STE 
value of each frame in the window, features listed below are extracted:
- Difference between frames with ZCR value above and below the mean
- Number of frames with ZCR value exceeding the threshold
- Third central moment about the mean of ZCR
- Standard deviation of first order difference of ZCR
- Modified Low Energy Ratio [1]
 

##
[1] E. Scheirer and M. Slaney, "Construction and evaluation of a robust multifeature speech/music discriminator," 1997 IEEE International Conference on Acoustics, Speech, and Signal Processing, Munich, 1997, pp. 1331-1334 vol.2.