import os
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "4" #please put your GPU

all_pic = sorted(os.listdir('./anime_pic/'), key=lambda x:int(x.split(".")[0]))

anime = pd.read_csv('anime_info.csv')
empty_lst = anime[(anime['pic_url'].apply(type) == float) | (anime['pic_url'] == 'None')]['MAL_ID'].to_list()
empty_lst = [f'{i}.jpg' for i in empty_lst]

genre = set()
for i in anime['Genders']:
    for j in i.split(', '):   
        genre.add(j)
        
genre = list(genre)
genre_dic = {genre[i]:i for i in range(len(genre))} 
total_genre = len(genre)

ids = []
labels = []
label = [0] * total_genre
pic_lst = []
count = 1
total = len(all_pic)

for pic in all_pic:
    print(f'\rprocessing {count}/{total} picture..',end='')
    if pic not in empty_lst:
        img = Image.open(f'anime_pic/{pic}').resize((125, 185))
        anime_id = pic[:-4]
        ids.append(anime_id)
        pic_lst.append(np.array(img))
        label = [0] * total_genre
        for i in anime[anime['MAL_ID'] == int(anime_id)]['Genders'].item().split(', '):
            label[genre_dic[i]] = 1
        labels.append(label)
    count += 1

features = np.stack(pic_lst,axis=0)

print('\nanime picture processing done!')

model = Sequential()  
inception = InceptionV3(input_shape = (225,335,3), weights='imagenet', include_top=False)
model.add(inception)
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=total_genre, activation="sigmoid"))
model.summary()

batchsize = 64

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

train_generator = datagen.flow(
    x = features,
    y = np.array(labels),
    shuffle = True,
    batch_size=batchsize,
    subset='training') # set as training data

validation_generator = datagen.flow(
    x = features,
    y = np.array(labels),
    batch_size=batchsize,
    subset='validation') # set as validation data

optim = 'adam'
model.compile(loss='binary_crossentropy',
              optimizer=optim,
              metrics=["binary_accuracy"])

checkpoint_save_path = 'anime_classification_less_dim.h5'
if os.path.exists(checkpoint_save_path):
    print('---------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

checkpoint = ModelCheckpoint(filepath=checkpoint_save_path, mode='max', monitor='val_binary_accuracy', verbose=0, save_best_only=True) 

epochs = 5

early_stop = EarlyStopping(monitor = 'loss', verbose = 0, patience = 5, mode = 'auto', restore_best_weights = True)

model.fit(train_generator,
          steps_per_epoch=train_generator.n/train_generator.batch_size ,
          epochs=epochs,
          validation_data=validation_generator,
          validation_steps=validation_generator.n/validation_generator.batch_size,
          verbose=1,
          callbacks=[checkpoint,early_stop])

datagen_output = ImageDataGenerator(rescale=1./255)

data_output = datagen_output.flow(x = features, y = np.array(labels), shuffle = False, batch_size=batchsize)

layer_name = 'dense_1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data_output)

with open('anime_image_features_less.npy', 'wb') as f:
    np.save(f, intermediate_output)
    
with open('anime_image_ids_less.npy', 'wb') as f:
    np.save(f, np.array(ids))

