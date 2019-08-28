'''文本分类

会将文本形式的影评分为“正面”或“负面”影评。这是一个二元分类（又称为两类分类）的示例，也是一种重要且广泛适用的机器学习问题。

'''

import tensorflow as tf
from tensorflow import keras

import numpy as np

imdb = keras.datasets.imdb

(train_data,train_lables),(test_data,test_label) = imdb.load_data(num_words = 10000)

print(train_data[0])

word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items() }
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i,'?') for i in text])


#sequences: List of lists, where each element is a sequence.
#maxlen: Int, maximum length of all sequences.
#dtype: Type of the output sequences.
#    To pad sequences with variable length strings, you can use `object`.
#padding: String, 'pre' or 'post':
#    pad either before or after each sequence.
#truncating: String, 'pre' or 'post':
#    remove values from sequences larger than
#    `maxlen`, either at the beginning or at the end of the sequences.
#value: Float or String, padding value.

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                       value = word_index["<PAD>"],
                                                       padding = 'post',
                                                       maxlen = 256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                      value = word_index["<PAD>"],
                                                      padding = 'post',
                                                      maxlen = 256)
###********构建模型*******######
##_________________________________________________________________
##Layer (type)                 Output Shape              Param #
##=================================================================
##embedding (Embedding)        (None, None, 16)          160000
##_________________________________________________________________
##global_average_pooling1d (Gl (None, 16)                0
##_________________________________________________________________
##dense (Dense)                (None, 16)                272
##_________________________________________________________________
##dense_1 (Dense)              (None, 1)                 17
##=================================================================
##Total params: 160,289
##Trainable params: 160,289
##Non-trainable params: 0

vocab_size = 10000
modle = keras.Sequential([
    keras.layers.Embedding(vocab_size,16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16,activation = tf.nn.relu),
    keras.layers.Dense(1,activation = tf.nn.sigmoid)
    ])

modle.summary()

modle.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])


x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_lables[:10000]
partial_y_train = train_lables[10000:]

history = modle.fit(partial_x_train,
                    partial_y_train,
                    epochs = 40,
                    batch_size = 512,
                    validation_data = (x_val,y_val),
                    verbose = 1)


results = modle.evaluate(test_data,test_label)
print(results)
history_dict = history.history
history_dict.keys()

#dict_keys(['loss','val_loss','val_acc','acc'])
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plt.plot(epochs,loss,'bo',label = 'training loss')
plt.plot(epochs,val_loss,'b',label = 'validation loss')
plt.title('train and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.show()