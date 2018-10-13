import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from pydub import AudioSegment
import random
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
from statistics import mode

#to try with different audio just change the filepath in the below function then runall
song = AudioSegment.from_wav("s1.wav")

#to randomly select 18.125 sec of audio from music data
end = random.randint(13593,len(song))
song_part_for_analysis = song[end-13593:end]

#save that segment of music in wav format
song_part_for_analysis.export("test.wav", format="wav")

def to_img_to_array(filename,data):
    sig, fs = librosa.load(filename,sr=44100)   
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    plt.figure(num = None,figsize = (4,3),dpi = 160)
    fig = librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    a = fig.get_figure()
    a.tight_layout(pad=0.0)
    a.canvas.draw()
    data = np.fromstring(a.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(a.canvas.get_width_height()[::-1] + (3,))
    plt.close() 
    width , height = 290,480
    img2 = []
    for counter in range(29):
        area = (width/29*counter , 0, width/29*(counter+1) , height)
        img = Image.fromarray(data, 'RGB')
        img = img.crop(area)
        img = np.asarray(img)
        img2.append(img)
    data = img2     
    return data

image = []
image = to_img_to_array('./test.wav',image)
image = np.asarray(image)
image = image.astype('float32')/255

#structure of my cnn network
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same',
                activation = 'relu', input_shape = (480,10,3)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.1))

model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.1))

model.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 128, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 1))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.3))

model.add(Dense(10, activation = 'softmax'))


#load our trained model weights
model.load_weights('model6.hdf5')

y_hat = model.predict(image)
mus_labels = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
result = []
for a in range(29):
    result.append(np.argmax(y_hat[a]))
print(mus_labels[mode(result)])    