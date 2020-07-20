# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from io import open
from itertools import islice

userDictPath = 'toClean/'


def main():
    dictFiles = []
    for (dirpath, dirnames, filenames) in os.walk(userDictPath):
        dictFiles.extend(filenames)
        break
    for dict in dictFiles:
        dict = userDictPath + dict
        names = set()
        with open(dict, 'r', encoding='utf-8') as fin:
            for line in islice(fin, 0, None):
                name = line.strip('\n')
                names.add(name)

        name_list = sorted(names)
        with open(dict, "w") as fout:
            fout.write("")
        with open(dict, "a", encoding='utf-8') as fout:
            for name in name_list:
                fout.write(name + "\n")


if __name__ == "__main__":
    main()
