import numpy as np
import os
import random

def newsample(nnn,ratio):
    if ratio > len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1), ratio)
    else:
        return random.sample(nnn, ratio)


def load_matrix(embedding_path,word_dict):
    print('load_matrix:')
    embedding_matrix = np.zeros((len(word_dict)+1, 300))
    have_word=[]
    with open(os.path.join(embedding_path,'glove.840B.300d.txt'),'rb') as f:
        while True:
            l=f.readline()
            if len(l)==0:
                break
            l=l.split()
            word = l[0].decode()
            if word in word_dict:
                index = word_dict[word]
                tp = [float(x) for x in l[1:]]
                embedding_matrix[index]=np.array(tp)
                have_word.append(word)
    return embedding_matrix,have_word


