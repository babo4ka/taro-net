from cards_loader import load_descs


descs = load_descs('general')

uniqueWords = []

for word in descs:
    if not word in uniqueWords:
        uniqueWords.append(word)

indexes_to_words = {}
words_to_indexes = {}

for i, word in enumerate(uniqueWords):
    indexes_to_words[i] = word
    words_to_indexes[word] = i
