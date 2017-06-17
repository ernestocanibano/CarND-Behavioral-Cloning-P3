import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing import image as keras_image

from random import uniform
from random import randint

# Funtions for data augmentation for the second track

def random_shadow(image, bright=uniform(0.2,1.0)):
    """
    Generate from 1 to 10 shadows in the image.
    :param image: image to modify
    :param bright: The bright of the shadows. If not it is generated randomly. 
    :return: Modified image
    """ 
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    n_shadows=randint(1,10) 
    
    for i in np.arange(1,n_shadows):
        x_ini = randint(0, hls.shape[1])
        y_ini = randint(0, hls.shape[0])
        x_fin = randint(x_ini,x_ini+hls.shape[1])
        y_fin = randint(y_ini,y_ini+int(hls.shape[0]/4))
        hls[y_ini:y_fin,x_ini:x_fin,1] = hls[y_ini:y_fin,x_ini:x_fin,1]*bright

    image = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
        
    return image


def random_brightness(image, bright=uniform(0.3,1.0)):  
    """
    Modify the brightness of the image.
    :param image: image to modify
    :param bright: Brightness. If not it is generated randomly. 
    :return: Modified image
    """ 
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
       
    hls[:,:,1] = hls[:,:,1]*bright
    
    image = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
    
    return image
	
# Functions to extract data from the records of the simulator

def extractFilesFromCsv(csv_path, lateralCams=True, correction=0.25):
    """
    Read the .csv file and extract the path of all the files.
    :param csv_path: path of the .csv file
    :param lateralCams: If true lateral cams images ar procesed
    :param correction: angle correction for lateral cams
    :return: A table with the paths of the files and their values: angle, throtlee
    """
    table=[]
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        path = csv_path.rsplit('/',1)[0] + '/IMG/'
        for line in reader:
            center = path+line[0].split('\\')[-1]
            left = path+line[1].split('\\')[-1]
            right = path+line[2].split('\\')[-1]
            steer = float(line[3])
            throtle = float(line[4])
            brake = float(line[5])
            speed = float(line[6])
          
            table.append([center, 0, steer, throtle, brake, speed])
            table.append([center, 4, steer, throtle, brake, speed]) #shadows
            table.append([center, 5, steer, throtle, brake, speed]) #brightness
            table.append([center, 1, -steer, throtle, brake, speed]) #flipped
            if lateralCams == True:
                table.append([left, 2, steer+correction, throtle, brake, speed])
                table.append([right, 3, steer-correction, throtle, brake, speed])
    return table

def generator(table, batch_size=32):
    """
    Split in batches and creates a generator to reduce the memory consumption during thr process of train the model
    :param table: table with the paths and values
    :param batch_size: batch_size
    :return: array of images and angle values to be used for trainning the model
    """ 
    num_files = len(table)
    while 1:
        table = shuffle(table)
        for offset in range(0, num_files, batch_size):
            fields = table[offset:offset+batch_size]
            images=[]
            measurements=[]
            for field in fields:
                measurements.append(float(field[2]))
                image = cv2.imread(field[0]) 
                if int(field[1]) == 1:
                    image = np.fliplr(image)
                if int(field[1]) == 4:
                    image = random_shadow(image)
                if int(field[1]) == 5:
                    image = random_brightness(image)                    
                images.append(image)  
            yield shuffle(np.array(images), np.array(measurements))

BATCH_SIZE = 32

track1_forward = extractFilesFromCsv('./records/track1_forward/driving_log.csv')
track1_reverse = extractFilesFromCsv('./records/track1_reverse/driving_log.csv')
track1_recovering = extractFilesFromCsv('./records/track1_recovering/driving_log.csv',False)
track2 = extractFilesFromCsv('./records/track2/driving_log.csv',True)
track1_forward.extend(track1_reverse)
track1_forward.extend(track1_recovering)
track1_forward.extend(track2)
track = shuffle(track1_forward)
table_train, table_valid = train_test_split(track, test_size=0.20)
print(len(table_train))
print(len(table_valid))

train_generator = generator(table_train, batch_size=BATCH_SIZE)
valid_generator = generator(table_valid, batch_size=BATCH_SIZE)

samples_per_epoch  = len(table_train)
nb_val_samples = len(table_valid)

### Model NVIDIA
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

model = Sequential()
# Normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss',patience=3,mode='auto')


history_object = model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch,  validation_data=valid_generator, nb_val_samples=nb_val_samples, nb_epoch=20, callbacks=[early_stopping])
model.save('model2_d100.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
    