# -*- coding: utf-8 -*-

from keras import models
from keras import layers
from keras import optimizers

import numpy as np

# States:
# 0 - unknown
# 1 - discard tsumogiri
# 2 - discard from hand
# 3 - discard after meld
# 4 - used in open set
# 5 - is dora - unused for now

# input - field state as 2D vector of [tile, state]
# output - vector [136]: 1 if oppent wait for this tile, 0 if he doesn't

samples = 1000
tiles = 136
states = 5

train_input_raw = [[[0 for x in range(states)] for y in range(tiles)] for z in range(samples)]
train_output_raw = [[0 for y in range(tiles)] for z in range(samples)]
test_input_raw = [[[0 for x in range(states)] for y in range(tiles)] for z in range(samples)]
test_output_raw = [[0 for y in range(tiles)] for z in range(samples)]

train_input = np.asarray(train_input_raw).astype('float32')
train_output = np.asarray(train_output_raw).astype('float32')
test_input = np.asarray(test_input_raw).astype('float32')
test_output = np.asarray(test_output_raw).astype('float32')

train_input = train_input.reshape((samples, tiles * states))
test_input = test_input.reshape((samples, tiles * states))

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(tiles * states,)))
model.add(layers.Dense(tiles, activation='softmax'))

model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

model.fit(train_input, train_output, epochs=20, batch_size=512)

test_loss, test_acc = model.evaluate(test_input, test_output)
