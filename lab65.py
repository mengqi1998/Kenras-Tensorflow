
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data()

word_index = imdb.get_word_index()
reverse_word_index = dict([(v, k) for (k, v) in word_index.items()])

# get first sentence
print(train_data[0])

for j in range(5):
    decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[j]])
    print(decoded_review)
print(train_labels[:5])