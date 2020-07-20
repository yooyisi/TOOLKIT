import random


def balance_train_by_sampling(train_clas_sample_dict, num):
    train_data, labels = [], []
    for k, v in train_clas_sample_dict.items():
        for i in range(num):
            train_data.append(random.choice(v))
            labels.append(k)
    return train_data, labels