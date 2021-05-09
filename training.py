import keras
from keras.layers import Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dense
from keras.models import Sequential,load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import sys

def train():
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory(
            'train',
            target_size=(150,150),
            batch_size=20,
            class_mode='binary')

    test_set = test_datagen.flow_from_directory(
            'test',
            target_size=(150,150),
            batch_size=20,
            class_mode='binary')

    model=Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
    model.add(MaxPooling2D() )
    model.add(Conv2D(32,(3,3),activation='relu'))
    model.add(MaxPooling2D() )
    model.add(Conv2D(32,(3,3),activation='relu'))
    model.add(MaxPooling2D() )
    model.add(Conv2D(32,(3,3),activation='relu'))
    model.add(MaxPooling2D() )
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

    try:
        early_stopping = sys.argv[2]
    except:
        early_stopping = False

    if early_stopping:
        fitted_model=model.fit(
            training_set,
            epochs=10,
            validation_data=test_set,
            callbacks=[es]
            )
    else:
        fitted_model=model.fit(
            training_set,
            epochs=10,
            validation_data=test_set
            )

    model.save('Fitted_Model.h5',fitted_model)

    return None