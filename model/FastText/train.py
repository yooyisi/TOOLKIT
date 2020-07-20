# -*- coding: utf-8 -*-
import fasttext
dev_sample_percentage = 0.2

# __label__真实标签\t分词后文本
model = fasttext.train_supervised('./corpus/train.txt')

model.predict("Which baking dish is best to bake a banana bread ?")
