import numpy as np

# ほぼゼロつく2と同じ。違いは日本語対応のためMeCabを使ったこと
def preprocess(text):
    import MeCab

    tagger = MeCab.Tagger("-Owakati")
    text = text.lower()
    words = tagger.parse(text).split()
    # print(words)

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word
