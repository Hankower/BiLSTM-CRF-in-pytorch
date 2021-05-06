import  pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from model import Model

def calculate(x,y,id2word,id2tag,res=[]):
    entity=[]
    for j in range(len(x)):
        if id2tag[y[j]]=='B':
            entity=[id2word[x[j]]]
        elif id2tag[y[j]]=='M' and len(entity)!=0:
            entity.append(id2word[x[j]])
        elif id2tag[y[j]]=='E' and len(entity)!=0:
            entity.append(id2word[x[j]])
            res.append(entity)
            entity=[]
        elif id2tag[y[j]]=='S':
            entity=[id2word[x[j]]]
            res.append(entity)
            entity=[]
        else:
            entity=[]
    return res


with open('../data/datasave.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)


TEST_DATA = "../data/test.txt"
SAVE_PATH = "../data/test_predict.txt"
wordnum = len(id2word)
x_test = []
with open(TEST_DATA, 'r', encoding="utf-8") as ifp:
    for line in ifp:
        line = line.strip()
        if not line:continue
        line_x = []
        for i in range(len(line)):
            if line[i] == " ":continue
            if(line[i] in id2word):
                line_x.append(word2id[line[i]])
            else:
                id2word.append(line[i])
                word2id[line[i]]=wordnum
                line_x.append(wordnum)
                wordnum=wordnum+1
        x_test.append(line_x)


model = torch.load("./model/model9.pkl")
result = ""
f_result = open(SAVE_PATH, 'w')


for sentence in x_test:
    tensor_s = torch.tensor(sentence, dtype=torch.long)
    score, predict = model.test(tensor_s)
    for i in range(len(predict)):
        result += id2word[sentence[i]]
        if id2tag[predict[i]] in ['S', 'E']:
            result += "  "
    result += "\n"


f_result.write(result)
f_result.close()
