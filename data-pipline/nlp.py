import jieba


def str2fenciStr(astr):
    return ' '.join(jieba.lcut(astr))
