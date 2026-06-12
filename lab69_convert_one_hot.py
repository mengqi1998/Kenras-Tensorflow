from keras.utils import to_categorical

origin_numbers = [1, 2, 5, 8, 9]
NUM_DIGIT = 10
converted = to_categorical(origin_numbers, NUM_DIGIT)
print(converted)

NUM_DIGIT2 = 20
converted2 = to_categorical(origin_numbers, NUM_DIGIT2)
print(converted2)