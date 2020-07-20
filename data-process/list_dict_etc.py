
def two_list2dict(key_list, val_list):
    dic = {}
    for a, b in zip(key_list, val_list):
        dic[a] = dic.get(a, [])
        dic[a].append(b)

    return dic


def list2countDict(alist):
    dic = {}
    for i in alist:
        dic[i] = dic.get(i, 0) + 1
    return dic


def sort_dic_byvalue(dic):
    """
    打印词典排序，sort dict
    :param dic:
    :return:
    """
    sorted_list = sorted(dic.items(), key=lambda x: x[1])
    for (k, v) in sorted_list:
        print(v, k)


