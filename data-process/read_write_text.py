import os
import pickle
from itertools import islice

sentlist = open('', 'r', encoding='utf-8').readlines()
sentlist = [i.strip('\n') for i in sentlist]



# XML
import xml.etree.ElementTree as ET
root = ET.parse('thefile.xml').getroot()

# pickle
with open('data.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

with open('data.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f)

with open(txtfile, 'r', encoding='utf-8') as fin:
    for line in islice(fin, 0, None):
        line = line.strip('\n')
        elements = line.split('\t')


# csv pandas
import pandas as pd
def csv_2_dict(filename, fieldnames, select_fields, keyname, delimiter=',', max_size=None):
    '''
    适合直接从db拿来的raw数据
    :param filename:
    :param fieldnames:
    :param select_fields:
    :param keyname:
    :return: 词典，value是list of dict：[{},{},..]
    '''
    fieldname_list = fieldnames.split(',')
    select_fields_list = select_fields.split(',')
    dictReader = pd.read_csv(filename, sep=delimiter, names=fieldname_list)
    # dictReader = csv.DictReader(codecs.open(filename, 'rU', encoding='utf-16'),
    #                             fieldnames=fieldname_list,
    #                             delimiter=delimiter, quotechar='"')
    is_first_line = True
    res = {}
    counter = 0
    for index, row in dictReader.iterrows():
        if is_first_line:
            is_first_line = False
            continue
        value_dict = {}
        for field_name in select_fields_list:
            if field_name == keyname:
                continue
            value_dict[field_name] = row[field_name]
        tmp_value_dict = res.get(row[keyname], [])
        tmp_value_dict.append(value_dict)
        res[row[keyname]] = tmp_value_dict
        counter += 1
        if counter % 10000 == 0:
            print(counter)
        if max_size:
            if counter > max_size:
                break

    return res

# read_all_txts under a dir and save in dict


userDictPath = 'toClean/'


def walk_all_txts(userDictPath):
    res = {}
    dictFiles = []
    for (dirpath, dirnames, filenames) in os.walk(userDictPath):
        dictFiles.extend(filenames)
        break
    for filename in dictFiles:
        dict = userDictPath + filename
        names = set()
        with open(dict, 'r', encoding='utf-8') as fin:
            for line in islice(fin, 0, None):
                name = line.strip('\n')
                names.add(name)
        res[filename] = names
    return res