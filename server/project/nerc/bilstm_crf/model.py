import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import autograd

import _pickle as cPickle
import numpy as np
import json
from collections import OrderedDict

import os
import sys
import codecs
import re
import numpy as np

from flask import Flask, jsonify
from flask_ngrok import run_with_ngrok
from flask import request


import sklearn_crfsuite
from sklearn_crfsuite import metrics
from seqeval.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')
def set_params():
    parameters = OrderedDict()
    parameters['train'] = data_loc + "data/eng.train" #Path to train file
    parameters['test'] = data_loc + "data/eng.testb" #Path to test file
    parameters['tag_scheme'] = "iob"
    parameters['lower'] = True
    parameters['zeros'] =  True
    parameters['char_dim'] = 30
    parameters['word_dim'] = 50
    parameters['word_lstm_dim'] = 100
    parameters['word_bidirect'] = True
    parameters['all_emb'] = 1
    parameters['crf'] =1
    parameters['dropout'] = 0.5
    parameters['gradient_clip']=5.0
    parameters['use_gpu'] = torch.cuda.is_available()
    return parameters

def get_pickle(name):
    file_path = data_dir + name + ".pkl"
    file_obj = open(file_path, 'rb')
    return cPickle.load(file_obj)

def lower_case(x,lower=False):
    return x.lower() if lower else x

def pickle_me(name, obj):
    file_path = data_dir + name + ".pkl"
    file_obj = open(file_path, 'wb')
    cPickle.dump(obj, file_obj)
    file_obj.close()
    print("Pickled the obj {} at {}".format(name, file_path))
    return
def forward_calc(self, sentence, chars, chars2_length, d):
    feats = self._get_lstm_features(sentence, chars, chars2_length, d)
    if self.use_crf:
        score, tag_seq = self.viterbi_decode(feats)
    else:
        score, tag_seq = torch.max(feats, 1)
        tag_seq = list(tag_seq.cpu().data)
    return score, tag_seq

def get_neg_log_likelihood(self, sentence, tags, chars2, chars2_length, d):
    feats = self._get_lstm_features(sentence, chars2, chars2_length, d)

    if self.use_crf:
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score
    else:
        tags = Variable(tags)
        scores = nn.functional.cross_entropy(feats, tags)
        return scores

def viterbi_algo(self, feats):
    backpointers = []
    init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
    init_vvars[0][self.tag_to_ix[START_TAG]] = 0
    forward_var = Variable(init_vvars)
    if self.use_gpu:
        forward_var = forward_var.cuda()
    for feat in feats:
        next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
        _, bptrs_t = torch.max(next_tag_var, dim=1)
        bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
        next_tag_var = next_tag_var.data.cpu().numpy()
        viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
        viterbivars_t = Variable(torch.FloatTensor(viterbivars_t))
        if self.use_gpu:
            viterbivars_t = viterbivars_t.cuda()
        forward_var = viterbivars_t + feat
        backpointers.append(bptrs_t)

    terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
    terminal_var.data[self.tag_to_ix[STOP_TAG]] = -10000.
    terminal_var.data[self.tag_to_ix[START_TAG]] = -10000.
    best_tag_id = argmax(terminal_var.unsqueeze(0))
    path_score = terminal_var[best_tag_id]

    best_path = [best_tag_id]
    for bptrs_t in reversed(backpointers):
        best_tag_id = bptrs_t[best_tag_id]
        best_path.append(best_tag_id)

    start = best_path.pop()
    assert start == self.tag_to_ix[START_TAG]
    best_path.reverse()
    return path_score, best_path

def forward_alg(self, feats):
    init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
    init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
    forward_var = autograd.Variable(init_alphas)
    if self.use_gpu:
        forward_var = forward_var.cuda()
    for feat in feats:
        emit_score = feat.view(-1, 1)
        tag_var = forward_var + self.transitions + emit_score
        max_tag_var, _ = torch.max(tag_var, dim=1)
        tag_var = tag_var - max_tag_var.view(-1, 1)
        forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1) # ).view(1, -1)
    terminal_var = (forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]).view(1, -1)
    alpha = log_sum_exp(terminal_var)
    return alpha

def get_lstm_features(self, sentence, chars2, chars2_length, d):
    chars_embeds = self.char_embeds(chars2).unsqueeze(1)
    chars_cnn_out3 = self.char_cnn3(chars_embeds)
    chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
                                         kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)
    embeds = self.word_embeds(sentence)
    embeds = torch.cat((embeds, chars_embeds), 1)
    embeds = embeds.unsqueeze(1)
    embeds = self.dropout(embeds)
    lstm_out, _ = self.lstm(embeds)
    lstm_out = lstm_out.view(len(sentence), self.hidden_dim*2)
    lstm_out = self.dropout(lstm_out)
    lstm_feats = self.hidden2tag(lstm_out)
    return lstm_feats

def score_sentences(self, feats, tags):
    r = torch.LongTensor(range(feats.size()[0]))
    if self.use_gpu:
        r = r.cuda()
        pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag_to_ix[STOP_TAG]])])
    else:
        pad_start_tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_to_ix[STOP_TAG]])])

    score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])
    return score

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                 char_to_ix=None, pre_word_embeds=None, char_out_dimension=25,char_embedding_dim=25, use_gpu=False
                 , use_crf=True):
        super(BiLSTM_CRF, self).__init__()
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.out_channels = char_out_dimension

        if char_embedding_dim is not None:
            self.char_embedding_dim = char_embedding_dim
            self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
            init_embedding(self.char_embeds.weight)
            self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(3, char_embedding_dim), padding=(2,0))
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False
        self.dropout = nn.Dropout(parameters['dropout'])
        self.lstm = nn.LSTM(embedding_dim+self.out_channels, hidden_dim, bidirectional=True)
        init_lstm(self.lstm)
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)
        init_linear(self.hidden2tag)
        if self.use_crf:
            self.transitions = nn.Parameter(
                torch.zeros(self.tagset_size, self.tagset_size))
            self.transitions.data[tag_to_ix[START_TAG], :] = -10000
            self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
    _score_sentence = score_sentences
    _get_lstm_features = get_lstm_features
    _forward_alg = forward_alg
    viterbi_decode = viterbi_algo
    neg_log_likelihood = get_neg_log_likelihood
    forward = forward_calc


def get_chunk_type(tok, idx_to_tag):
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def get_chunks(seq, tags):
    default = tags["O"]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        if tok == default and chunk_type is not None:
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def zero_digits(s):
    return re.sub('\d', '0', s)


def load_sentences(path, zeros):
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=False):
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[lower_case(w, lower) if lower_case(w, lower) in word_to_id else '<UNK>']
                 for w in str_words]
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'tags': tags,
        })
    return data


def eval_method(model, datas, dataset="Train"):
    y_truth = []
    y_pred = []
    prediction = []
    correct_preds, total_correct, total_preds = 0., 0., 0.
    for data in datas:
        # print(data)
        # return
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']
        d = {}
        chars2_length = [len(c) for c in chars2]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
        for i, c in enumerate(chars2):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = Variable(torch.LongTensor(chars2_mask))
        dwords = Variable(torch.LongTensor(data['words']))

        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, chars2_length, d)
        predicted_id = out
        """
        For micro and macro average
        id_to_tag
        """
        ground_truth_tags = []
        predicted_tags = []
        for ele in ground_truth_id:
            ground_truth_tags.append(id_to_tag[ele])
        for ele2 in predicted_id:
            predicted_tags.append(id_to_tag[ele2])
        y_truth.append(ground_truth_tags)
        y_pred.append(predicted_tags)
        """ ENDS here """
        lab_chunks = set(get_chunks(ground_truth_id, tag_to_id))
        lab_pred_chunks = set(get_chunks(predicted_id, tag_to_id))

        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    # Calculating the Precision, Recall, and F1-Score
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f_measure = 2 * p * r / (p + r) if correct_preds > 0 else 0

    print("{}: Precision: {} Recall: {} F-measure: {}".format(dataset, p, r, f_measure))
    return p, r, f_measure, y_truth, y_pred
    # return p, r, f_measure


def pred_wrapper(data_arg):
  data = (data_arg['data']).strip()
  sentences = [i.strip() for i in data.split(".") if i != '']
  res = predictor(sentences)
  ret_list = []
  ner_ctr = 0
  for ele in res:
      # 'NA' or 'O' tag
    key = list(ele.keys())[-1]
    val = list(ele.values())[-1]
    if val != 'O':
        ret_list.append([key, val])
        ner_ctr+=1
    #   retDict[list(ele.keys())[-1]] = list(ele.values())[-1]
#   print(retDict)
  return json.dumps({'count': ner_ctr, 'data': ret_list})


def predictor(model_testing_sentences):
    res = []
    lower = parameters['lower']
    final_test_data = []
    for sentence in model_testing_sentences:
        s=sentence.split()
        str_words = [w for w in s]
        words = [word_to_id[lower_case(w,lower) if lower_case(w,lower) in word_to_id else '<UNK>'] for w in str_words]
        chars = [[char_to_id[c] for c in w if c in char_to_id] for w in str_words]
        final_test_data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
        })
    predictions = []
    for data in final_test_data:
        words = data['str_words']
        chars2 = data['chars']
        d = {}
        chars2_length = [len(c) for c in chars2]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
        for i, c in enumerate(chars2):
            chars2_mask[i, :chars2_length[i]] = c
        chars2_mask = Variable(torch.LongTensor(chars2_mask))
        dwords = Variable(torch.LongTensor(data['words']))
        if use_gpu:
            val,predicted_id = model(dwords.cuda(), chars2_mask.cuda(), chars2_length, d)
        else:
            val,predicted_id = model(dwords, chars2_mask, chars2_length, d)
        # print(val, predicted_id)
        for word,tag_id in zip(words,predicted_id):
            tag_val = id_to_tag[tag_id]
            res.append(({word:tag_val}))
        # pred_chunks = get_chunks(predicted_id,tag_to_id)
        # temp_list_tags=['NA']*len(words)
        # for p in pred_chunks:
        #     temp_list_tags[p[1]]=p[0]
        # for word,tag in zip(words,temp_list_tags):
        #     res.append({word:tag})
    print("RES:", res)
    return res


data_dir = "/content/drive/Shareddrives/SWM - NER/models/BILSTM_CRF/"
data_loc = "/content/drive/Shareddrives/SWM/"


parameters = set_params()
use_gpu = parameters['use_gpu']
START_TAG = '<START>'
STOP_TAG = '<STOP>'

model = get_pickle('model')
char_to_id = get_pickle('char_to_id')
word_to_id = get_pickle('word_to_id')
tag_to_id = get_pickle('tag_to_id')

id_to_tag = get_pickle('id_to_tag')
print(id_to_tag)

train_sentences = load_sentences(parameters['train'], parameters['zeros'])
test_sentences = load_sentences(parameters['test'], parameters['zeros'])

train_data = prepare_dataset(train_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower'])
test_data = prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower'])


p, r, f_measure, y_true, y_pred = eval_method(model, train_data, dataset="Test")


all_tags = ['B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC','B-MISC','I-MISC', 'O']
labels = list(all_tags)
sorted_labels = sorted( labels, key=lambda name: (name[1:], name[0]))

print(metrics.flat_classification_report(
    y_true, y_pred, labels=sorted_labels, digits=3, ))

print("   micro avg     ", metrics.flat_f1_score(
    y_true, y_pred, labels=sorted_labels,average='micro'))
