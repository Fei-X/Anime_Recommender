import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import LSTM, Embedding, Dense, Bidirectional
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "2" #please put your GPU

anime = pd.read_csv('anime_info.csv')
anime = anime.drop(1411).drop_duplicates(subset=['sypnopsis']).reset_index(drop=True)
ids = anime['MAL_ID']

# Preprocessing
def preprocessing(df):
    df['text'] = df['sypnopsis'].str.lower()
    df['text'] = df['text'].str.replace(' [^\w\s]+',' ',regex=True)
    df['text'] = df['text'].str.replace('[^\w\s]+ ',' ',regex=True)
    df['text'] = df['text'].str.replace("'t",'t',regex=True)
    for item in ["'s","'re","'ve","'ll","´s","'d","´t","’s"]:
        df['text'] = df['text'].str.replace(item,'',regex=True)
    return df

anime = preprocessing(anime)
anime['text'].apply(lambda x : len(x.split(' ')) if isinstance(x, str) else 0).quantile(0.95)

genre = set()
for i in anime['Genders']:
    for j in i.split(', '):   
        genre.add(j)
        
genre = list(genre)
genre_dic = {genre[i]:i for i in range(len(genre))} 
total_genre = len(genre)

ids = []
syn_lst = []
labels = []

for i in anime['MAL_ID']:
    syn = anime[anime['MAL_ID'] == i]['text'].item()
    print(f'\rprocessing anime {i}',end='')
    if isinstance(syn, str):
        ids.append(i)
        syn_lst.append(syn)
        label = [0] * total_genre
        for j in anime[anime['MAL_ID'] == i]['Genders'].item().split(', '):
            label[genre_dic[j]] = 1
        labels.append(label)
print('\nprocessing done!')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(syn_lst)
tok = tokenizer.texts_to_sequences(syn_lst)

max_seq_len = 180
vec = pad_sequences(tok, maxlen=max_seq_len, padding='post')

vocabulary = len(tokenizer.word_counts) + 2 # add 2 because one for index 0 and one for unknown words
embedding_dim = 300
state_dim = 128

path_to_glove_file = './pretrained/glove.42B.300d.txt'

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

# The embedding matrix will be 300d for each word in vocabulary.
# If the word is not in Glove, it would be all 0.
hits = 0
misses = 0
# misses_lst = [] # investigating the missing words

# Prepare embedding matrix
embedding_matrix = np.zeros((vocabulary, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
        # misses_lst.append(word)
print("Converted %d words (%d misses)" % (hits, misses))

state_dim = 64
max_seq_len = 180

embedding_layer = Embedding(
    vocabulary,
    embedding_dim,
    input_length=max_seq_len,
    embeddings_initializer=Constant(embedding_matrix),
    trainable=False,
)

model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(state_dim, return_sequences=True, dropout=0.2)))
model.add(Bidirectional(LSTM(state_dim, return_sequences=True, dropout=0.2)))
model.add(Bidirectional(LSTM(state_dim, return_sequences=False, dropout=0.2)))
model.add(Dense(units=total_genre, activation="sigmoid"))
model.summary()

batchsize = 1024
epoches = 20


model.compile(optimizer=RMSprop(learning_rate=0.001),loss='binary_crossentropy',metrics=['acc'])

x_train_vec = vec[:12000]
y_train = np.array(labels[:12000])

x_valid_vec = vec[12000:]
y_valid = np.array(labels[12000:])

checkpoint_save_path = 'anime_text_classification.h5'
if os.path.exists(checkpoint_save_path):
    print('---------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
    
checkpoint = ModelCheckpoint(checkpoint_save_path, mode='auto', monitor='val_loss', verbose=0, save_best_only=True)

early_stop = EarlyStopping(monitor = 'loss', verbose = 0, patience = 5, mode = 'auto', restore_best_weights = True)

model.fit(x_train_vec,y_train, epochs=epoches, verbose=1,
          batch_size=128,validation_data=(x_valid_vec, y_valid),callbacks=[checkpoint, early_stop])

layer_name = 'bidirectional_2'
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(vec)

with open('anime_text_features.npy', 'wb') as f:
    np.save(f, intermediate_output)
    
with open('anime_text_ids.npy', 'wb') as f:
    np.save(f, np.array(ids))
