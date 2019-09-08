import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

NUM_WORD = 10000
(train_data,train_label),(test_data,test_label) = keras.datasets.imdb.load_data(num_words = NUM_WORD)

def multi_hot_sequence(sequence,dimension):
    results = np.zeros((len(sequence),dimension))
    for i , word_indices in enumerate(sequence):
        results[i,word_indices] = 1.0
    return results

train_data = multi_hot_sequence(train_data,dimension = NUM_WORD)
test_data = multi_hot_sequence(test_data,dimension = NUM_WORD)

baseline_model = keras.Sequential([
    keras.layers.Dense(16,activation=tf.nn.relu,input_shape=(NUM_WORD,)),
    keras.layers.Dense(16,activation=tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)
    ])

baseline_model.compile(optimizer='adam',
                       loss = 'binary_crossentropy',
                       metrics=['accuracy','binary_crossentropy'])
baseline_model.summary()

baseline_history = baseline_model.fit(train_data,
                                      train_label,
                                      epochs = 20,
                                      batch_size = 512,
                                      validation_data = (test_data,test_label),
                                      verbose =2)

##user smaller model
smaller_model = keras.Sequential([
    keras.layers.Dense(4,activation=tf.nn.relu,input_shape=(NUM_WORD,)),
    keras.layers.Dense(4,activation=tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)
    ])
smaller_model.compile(optimizer='adam',
                       loss = 'binary_crossentropy',
                       metrics=['accuracy','binary_crossentropy'])
smaller_model.summary()

smaller_history = smaller_model.fit(train_data,
                                    train_label,
                                    epochs = 20,
                                    batch_size = 512,
                                    validation_data=(test_data,test_label),
                                    verbose=2
                                    )

##user bigger model
bigger_model = keras.Sequential([
    keras.layers.Dense(512,activation=tf.nn.relu,input_shape=(NUM_WORD,)),
    keras.layers.Dense(512,activation=tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)
    ])
bigger_model.compile(optimizer='adam',
                       loss = 'binary_crossentropy',
                       metrics=['accuracy','binary_crossentropy'])

bigger_model.summary()

bigger_history = bigger_model.fit(train_data,
                                    train_label,
                                    epochs = 20,
                                    batch_size = 512,
                                    validation_data=(test_data,test_label),
                                    verbose=2
                                    )


def plot_history(historys,key= 'binary_crossentropy'):
   plt.figure(figsize =(16,10))
   for name,history in historys:
       val = plt.plot(history.epoch,history.history['val_'+key],
                      '--',label=name.title() + 'val')
       plt.plot(history.epoch,history.history[key],color =val[0].get_color(),label=name.title() + 'train')
   plt.xlabel('epochs')
   plt.ylabel(key.replace('_',' ').title())
   plt.legend()
   plt.xlim(0,max(history.epoch))
   plt.show()

plot_history([('baseline',baseline_history),
              ('smaller',smaller_history),
              ('bigger',bigger_history)])

l2_model = keras.models.Sequential([
    keras.layers.Dense(16,kernel_regularizer=keras.regularizers.l2(0.001),activation=tf.nn.relu,input_shape=(NUM_WORD,)),
    keras.layers.Dense(16,kernel_regularizer=keras.regularizers.l2(0.001),activation=tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)
    ])

l2_model.compile(optimizer='adam',
                 loss = 'binary_crossentropy',
                 metrics=['accuracy','binary_crossentropy'])

l2_model_history = l2_model.fit(train_data, train_labels,
                                epochs=20,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                verbose=2)


dpt_model = keras.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

dpt_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy','binary_crossentropy'])

dpt_model_history = dpt_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)

plot_history([('baseline',baseline_history),
              ('l2model',l2_model_history),
              ('dptmodel',dpt_model_history)])