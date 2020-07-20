# -*- coding: utf-8 -*-
# @Time: 2020/4/29 15:51
# @Author:
import random

file_path = 'to_shuffle.txt'
lines = open(file_path, 'r', encoding='utf-8').readlines()
random.shuffle(lines)
open(file_path, 'w', encoding='utf-8').writelines(lines)
