
from flask import request
import json
import warnings

from project import app
from project.nerc.crf import model

warnings.filterwarnings('ignore')


mapping = {'0': 'O',
           '1': 'B-PER',
           '2': 'I-PER',
           '3': 'B-ORG',
           '4': 'I-ORG',
           '5': 'B-LOC',
           '6': 'I-LOC',
           '7': 'B-MISC',
           '8': 'I-MISC'
           }


def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }




def pos_tag(sentence):
    sentence_features = [features(sentence, index) for index in range(len(sentence))]
    return list(zip(sentence, model.predict([sentence_features])[0]))


def crf_predict(data_arg):
    data = data_arg['data'].replace(".", " ")
    sentences = [i.strip() for i in data.split(" ") if i != '']
    res = pos_tag(sentences)
    ret_list = []
    ner_ctr = 0
    for word, tag in res:
        # print(tag)
        # print(type(tag))
        if tag != '0':
            ret_list.append([word, mapping[tag]])
            ner_ctr += 1
    return json.dumps({'count': ner_ctr, 'data': ret_list})
