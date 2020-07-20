import collections

import gensim
from torchtext.vocab import Vectors, Vocab

model = gensim.models.KeyedVectors.load_word2vec_format('input/vector.bin', binary=True)
print(model['中国'])

# 肉眼可读方式存储的word2vec
vectors = Vectors(word_vector, cache=wv_path)
vocab = Vocab(collections.Counter(words), vectors=vectors, specials=['<pad>', '<unk>'], min_freq=1)
wv_size = vocab.vectors.size()
vocab.stoi['<unk>']
