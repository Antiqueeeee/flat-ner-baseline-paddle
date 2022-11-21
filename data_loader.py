import json


import paddle
import numpy as np
import prettytable as pt
from paddlenlp.transformers import BertTokenizer
import warnings
import os
warnings.filterwarnings('ignore')

class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'
    def __init__(self):
        # self.label2id, self.id2label = {self.PAD : 0},{0 : self.PAD}
        self.label2id, self.id2label = {}, {}
    def add_label(self, label):
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label
        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)
    def label_to_id(self, label):
        return self.label2id[label]
    def id_to_label(self, i):
        return self.id2label[i]
    def save_Vocabulary(self,save_path):
        label2id_path = os.path.join(save_path,"label2id.json")
        id2label_path = os.path.join(save_path,"id2label.json")
        with open(label2id_path,"w",encoding="utf-8") as f:
            json.dump(self.label2id,f,ensure_ascii=False,indent=2)
        with open(id2label_path,"w",encoding="utf-8") as f:
            json.dump(self.id2label,f,ensure_ascii=False,indent=2)
    def load_Vocabulary(self,save_path):
        label2id_path = os.path.join(save_path, "label2id.json")
        id2label_path = os.path.join(save_path, "id2label.json")
        with open(label2id_path,"r",encoding="utf-8") as f:
            self.label2id = json.load(f)
        with open(id2label_path,"r",encoding="utf-8") as f:
            self.id2label = json.load(f)

class NERDataset(paddle.io.Dataset):
    def __init__(self, bert_inputs, bert_labels, sent_length):
        super(NERDataset,self).__init__()
        self.bert_inputs = bert_inputs
        self.bert_labels = bert_labels
        self.sent_length = sent_length

    def __getitem__(self, item):
        item_inputs = self.bert_inputs[item]
        item_labels = np.array(self.bert_labels[item])
        item_length = self.sent_length[item]
        return item_inputs,item_labels,item_length

    def __len__(self):
        return len(self.bert_inputs)
class RealDataset(paddle.io.Dataset):
    def __init__(self, bert_inputs, sent_length):
        super(RealDataset,self).__init__()
        self.bert_inputs = bert_inputs
        self.sent_length = sent_length

    def __getitem__(self, item):
        return self.bert_inputs[item],self.sent_length[item]

    def __len__(self):
        return len(self.bert_inputs)


def fill_vocab(vocab, dataset):
    entity_num = 0
    for data_item in dataset:
        for tag in data_item["label"]:
            vocab.add_label(tag)
            if "B-" in tag:
                entity_num += 1
    return entity_num
def load_data_bert(config):
    with open('./data/{}/train.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/{}/dev.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    tokenizer = BertTokenizer.from_pretrained(config.bert_name)
    vocab = Vocabulary()
    train_ent_num = fill_vocab(vocab, train_data)
    dev_ent_num = fill_vocab(vocab, dev_data)

    table = pt.PrettyTable([config.dataset, 'sentences', 'entities'])
    table.add_row(['train', len(train_data), train_ent_num])
    table.add_row(['dev', len(dev_data), dev_ent_num])
    config.logger.info("\n{}".format(table))
    config.label_num = len(vocab.label2id)
    vocab.save_Vocabulary(os.path.join("./outputs",config.dataset))
    config.vocab = vocab

    train_dataset = NERDataset(*process_bert(train_data, tokenizer, vocab))
    dev_dataset = NERDataset(*process_bert(dev_data, tokenizer, vocab))

    return (train_dataset, dev_dataset), (train_data, dev_data)
def load_real_bert(args):
    #vocab中相关信息没有保存，还是需要重新读取一遍数据
    with open('./data/{}/test.json'.format(args.task), 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    tokenizer = BertTokenizer.from_pretrained(args.bert_name)
    vocab = Vocabulary()
    vocab.load_Vocabulary(os.path.join("./outputs",args.task))
    args.vocab = vocab
    args.label_num = len(vocab.label2id)
    real_dataset = RealDataset(*process_real(test_data, tokenizer, vocab))
    return real_dataset, test_data
def process_bert(data, tokenizer, vocab):
    bert_inputs = list()
    instance_labels = list()
    sent_length = list()
    for index, instance in enumerate(data):
        tokens = [tokenizer.tokenize(word) for word in instance["sentence"]]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])
        instance["label"] = ["O"] + instance["label"] + ["O"]
        length = len(instance["label"])
        bert_labels = [vocab.label_to_id(tag) for tag in instance["label"]]
        bert_inputs.append(_bert_inputs)
        instance_labels.append(bert_labels)
        sent_length.append(length)
    return bert_inputs, instance_labels, sent_length
def process_real(data, tokenizer, vocab):
    bert_inputs = list()
    sent_length = list()

    for index, instance in enumerate(data):
        tokens = [tokenizer.tokenize(word) for word in instance["sentence"]]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])
        length = len(_bert_inputs)
        bert_inputs.append(_bert_inputs)
        sent_length.append(length)
    return bert_inputs, sent_length
def collate_fn(data):
    _bert_inputs, _bert_labels, sent_length = map(list, zip(*data))
    batch_size = len(_bert_inputs)
    bert_inputs = paddle.zeros((batch_size,max(sent_length)),dtype="int64")
    bert_labels = paddle.zeros((batch_size,max(sent_length)),dtype="int64")
    def fill(data,new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0]] = x
        return new_data
    bert_inputs = fill(_bert_inputs, bert_inputs)
    bert_labels = fill(_bert_labels, bert_labels)
    return bert_inputs, bert_labels, sent_length
def pred_collate_fn(data):
    _bert_inputs, sent_length = map(list, zip(*data))
    batch_size = len(_bert_inputs)
    bert_inputs = paddle.zeros((batch_size,max(sent_length)),dtype="int64")
    def fill(data,new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0]] = x
        return new_data
    bert_inputs = fill(_bert_inputs, bert_inputs)
    return bert_inputs, sent_length