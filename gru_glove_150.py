# import libs
import numpy as np
import pandas as pd
import pickle

from keras.callbacks import Callback, History
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from keras.layers import Dense, Input, GRU, Bidirectional, Conv1D
from keras.layers import Dropout, Embedding
from keras.preprocessing import text, sequence
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from keras.models import Model

# import files
EMBEDDING_FILE = "glove/toxic_vectors_150.txt"
train = pd.read_csv('data/clean_train.csv')
test = pd.read_csv('data/clean_test.csv')

# constants
max_features=100000
maxlen=150
embed_size=150

# data
classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train['comment_text'].fillna(' ', inplace=True)
test['comment_text'].fillna(' ', inplace=True)

train_y = train[classes].values
train_x = train['comment_text'].str.lower()
test_x = test['comment_text'].str.lower()

# Vectorize text + Embedding
tokenizer = text.Tokenizer(num_words=max_features, lower=True)
tokenizer.fit_on_texts(train_x.values)

train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)

train_x = sequence.pad_sequences(train_x, maxlen=maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=maxlen)

embeddings_index = {}
with open(EMBEDDING_FILE, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

word_index = tokenizer.word_index
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print("data embedded")

# Model setup

# Build Model
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
x = SpatialDropout1D(0.35)(x)
x = Bidirectional(GRU(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])
out = Dense(6, activation='sigmoid')(x)

model = Model(inp, out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

class NBatchLogger(Callback):
    """
    A Logger that log average performance per display steps.
    """
    def on_train_begin(self, logs={}):
        self.step = 0
        self.display = 1000
        self.metric_cache = {}
        self.losses = []
        self.accs = []
        self.logs = []

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ""
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)

                if k == 'loss':
                    self.losses.append(val)
                if k == 'acc':
                    self.accs.append(val)
            print('\nstep: {}/{} ... {}'.format(self.step,
                                          self.params['steps'],
                                          metrics_log))
            self.logs.append(metrics_log)
            self.metric_cache.clear()

class F1MetricsLogger(Callback):
    """
    A Logger that calculates F1-related metrics on the validation set.
    """
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targets = self.validation_data[1]
        
        f1 = f1_score(val_targets, val_predict, average='weighted')
        recall = recall_score(val_targets, val_predict, average='weighted')
        precision = precision_score(val_targets, val_predict, average='weighted')
        self.val_f1s.append(f1)
        self.val_recalls.append(recall)
        self.val_precisions.append(precision)
        print("f1: " + str(f1) + " precision: " + str(precision) + " recall: " + str(recall))
 
# training
batch_size = 32
epochs = 5
nbatch = NBatchLogger()
f1 = F1MetricsLogger()
history = History()

print("begin training")

hist = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[nbatch, f1, history], validation_split=0.1)
print(hist.history)
prefix = 'gru_glove_epoch_5_150d'
model.save(prefix + '.h5')
pickle.dump(nbatch.logs, open(prefix + '_nbatch.p', 'wb'))
pickle.dump(nbatch.losses, open(prefix + '_losses.p', 'wb'))
pickle.dump(nbatch.accs, open(prefix + '_accs.p', 'wb'))
pickle.dump(f1.val_f1s, open(prefix + '_f1s.p', 'wb'))
pickle.dump(f1.val_recalls, open(prefix + '_recalls.p', 'wb'))
pickle.dump(f1.val_precisions, open(prefix + '_precisions.p', 'wb'))