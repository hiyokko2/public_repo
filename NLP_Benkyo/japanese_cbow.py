# import MeCab

# tagger = MeCab.Tagger("-Owakati")

# result = tagger.parse("I have a pen. You have a dance.")

# print(result.split())


from my_utils import preprocess


# window_size = 1
# hidden_size = 5     # 中間層のサイズ（単語ベクトルの次元数）


japanese_text = "梅雨で雨の日が多いですね。早く梅雨が明けて欲しいですね。"
corpus, word_to_id, id_to_word = preprocess(japanese_text)
print(corpus)
print(word_to_id)
print(id_to_word)

model = SimpleCBOW()