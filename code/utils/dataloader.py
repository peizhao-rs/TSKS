import json
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


def pad_data(insts, pad_len, pad_num=-1, pad_idx=0):
    insts_pad = []
    if isinstance(insts[0], list):
        for inst in insts:
            inst_pad = inst + [pad_idx] * (pad_len - len(inst))
            insts_pad.append(inst_pad)
        if len(insts_pad) < pad_num:
            insts_pad += [[pad_idx] * pad_len] * (pad_num - len(insts_pad))
    else:
        insts_pad = insts + [pad_idx] * (pad_len - len(insts))
    return insts_pad


def len_to_mask(lens, _bool=False):
    batch_size = len(lens)
    max_len = max(lens)

    mask = np.zeros([batch_size, max_len])
    for i in range(batch_size):
        mask[i, :lens[i]] = 1.0
    if _bool:
        mask = mask.__eq__(0.0)
    return mask.tolist()


class MyDataSet(Dataset):
    def __init__(self, data_path, block_data=None):
        if block_data is None:
            self.data = self.read_json_data(data_path)
        else:
            self.data = block_data

    def read_json_data(self, data_path):
        with open(data_path, 'r', encoding='utf8') as fr:
            data = json.load(fr)
        return data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class SampleGenerator:
    def __init__(self,
                 data_path,
                 batch_size,
                 block_data = None
                 ):
        if block_data is None:
            self.loader = DataLoader(
                dataset=MyDataSet(data_path),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.process
            )
        else:
            self.loader = DataLoader(
                dataset=MyDataSet(None, block_data),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.process
            )
        
        self.data_size = self.loader.dataset.__len__()
    def process(self, batch):
        return batch
    def __call__(self):
        return self.loader


class DuconvLoader:
    def __init__(self,
                 vocab,
                 data_path,
                 batch_size,
                 shuffle=False
                 ):
        self.loader = DataLoader(
            dataset=MyDataSet(data_path),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.process
        )
        self.vocab = vocab
        self.pad_idx = vocab.word2id('<pad>')
        self.sos_idx = vocab.word2id('<sos>')
        self.eos_idx = vocab.word2id('<eos>')
        self.sep_idx = vocab.word2id('<sep>')

    def process(self, batch):

        batch_history_ids = []
        batch_response_ids = []
        batch_labels = []
        batch_kg_ids = []
        batch_kg_nums = []  # 记录每一个sample对应的知识数量
        batch_kg_len = []  # 记录每一条知识的长度

        batch_kg_extend_len = [] # 记录每一个sample对应的扩展词表长度
        batch_kg_extend_vocab = [] # 记录每一个sample对应的扩展词表
        batch_oovs = [] # 记录每一个sample对应的oov词汇

        for item in batch:
            # 取出goal三元组中的词，放在history前面，当对话历史为空的时候，作为历史信息
            goal_words = []
            for goal in item['goal']:
                goal_words.extend(goal[0])
                goal_words.extend(goal[1])
                goal_words.extend(goal[2])
            # 取history
            history_words = goal_words + ['<sep>']
            [history_words.extend(e) for e in item['history']]
            _, oovs = self.vocab.history2ids(history_words)
            batch_history_ids.append(
                self.vocab.words2ids(history_words)
            )

            # 取knowledge
            kg = item['knowledge']
            batch_kg_ids.append([])
            batch_kg_len.append([])
            batch_kg_extend_vocab.append([])
            batch_kg_extend_len.append([])
            for triple in kg:
                triple_words = triple[0] + triple[1] + triple[2]  # head, relation, tail 对应的词拼接成序列
                triple_ids = self.vocab.words2ids(triple_words)
                kg_extend_ids, oovs = self.vocab.kg2ids(triple_words, oovs)
                batch_kg_extend_vocab[-1].append(kg_extend_ids)
                batch_kg_extend_len[-1].append(len(kg_extend_ids))    
                batch_kg_ids[-1].append(triple_ids)
                batch_kg_len[-1].append(len(triple_ids))
            batch_oovs.append(oovs)
            batch_kg_nums.append(len(kg))
            # 取response
            batch_response_ids.append(
                self.vocab.words2ids(['<sos>'] + item['response'] + ['<eos>'])
            )
            batch_labels.append(self.vocab.response2ids(item['response'] + ['<eos>'], oovs))

        # padding
        # history
        len_history = [len(history) for history in batch_history_ids]
        pad_history_ids = pad_data(batch_history_ids, pad_len=max(len_history), pad_idx=self.pad_idx)
        history_mask = len_to_mask(len_history)

        # response & labels
        len_response = [len(response) for response in batch_response_ids]
        pad_response_ids = pad_data(batch_response_ids, pad_len=max(len_response), pad_idx=self.pad_idx)
        pad_labels = pad_data(batch_labels, pad_len=max(len_response)-1, pad_idx=self.pad_idx)

        # knowledge
        pad_kg_len = max(max(item) for item in batch_kg_len)
        pad_kg_num = max(batch_kg_nums)
        pad_kg_ids = [pad_data(item, pad_len=pad_kg_len, pad_num=pad_kg_num, pad_idx=self.pad_idx) for item in batch_kg_ids]
        kg_mask = len_to_mask(batch_kg_nums)  # for knowledge selection

        # extend knowledge vocab
        pad_kg_extend_len = max(max(item) for item in batch_kg_extend_len)
        pad_kg_extend_vocab = [pad_data(item, pad_len=pad_kg_extend_len, pad_num=pad_kg_num, pad_idx=self.pad_idx) for item in batch_kg_extend_vocab]

        # pad kg_len
        pad_batch_kg_len = pad_data(batch_kg_len, pad_len=pad_kg_num, pad_idx=1)  # 用GRU/LSTM编码，pack的时候，sample对应的seq_len不能为0
        len_kg = []
        [len_kg.extend(item) for item in pad_batch_kg_len]

        # kg_padding)mask
        kg_padding_mask = len_to_mask(len_kg)
        # max oov words num
        max_kg_oov_num = max([len(item) for item in batch_oovs])

        return pad_history_ids, len_history, history_mask, \
                pad_response_ids, len_response, pad_labels, \
                pad_kg_ids, len_kg, kg_mask, pad_kg_len, \
                pad_kg_extend_vocab, kg_padding_mask, max_kg_oov_num, batch_oovs

    def __call__(self):
        return self.loader


class BBCLoader:
    def __init__(self,
                 vocab,
                 data_path,
                 batch_size,
                 shuffle=False,
                 max_sent_len=40,
                 max_content_len=100
                 ):
        self.loader = DataLoader(
            dataset=MyDataSet(data_path),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.process
        )
        self.vocab = vocab
        self.pad_idx = vocab.word2id('<pad>')
        self.sos_idx = vocab.word2id('<sos>')
        self.eos_idx = vocab.word2id('<eos>')
        self.sep_idx = vocab.word2id('<sep>')
        self.max_sent_len = max_sent_len
        self.max_content_len = max_content_len

    def process(self, batch):
        batch_history_ids = []
        batch_response_ids = []
        batch_labels = []
        batch_kg_ids = []
        batch_kg_nums = []  # 记录每一个sample对应的知识数量
        batch_kg_len = []  # 记录每一条知识的长度

        batch_kg_extend_len = [] # 记录每一个sample对应的扩展词表长度
        batch_kg_extend_vocab = [] # 记录每一个sample对应的扩展词表
        batch_oovs = [] # 记录每一个sample对应的oov词汇

        
        for item in batch:

            history_words = []
            [history_words.extend(e.split()[:self.max_sent_len]) for e in item['history']]
            history_words = history_words[-self.max_content_len:]
            _, oovs = self.vocab.history2ids(history_words)
            response_words = item['response'].split()[:self.max_sent_len]

            batch_history_ids.append(
                self.vocab.words2ids(history_words)
            )
            batch_response_ids.append(
                self.vocab.words2ids(['<sos>'] + response_words + ['<eos>'])
            )


            # 取knowledge
            kg = item['knowledge']
            batch_kg_ids.append([])
            batch_kg_len.append([])
            batch_kg_extend_vocab.append([])
            batch_kg_extend_len.append([])
            for kg_item in kg:
                words = kg_item.split()[:self.max_sent_len]
                ids = self.vocab.words2ids(words)
                kg_extend_ids, oovs = self.vocab.kg2ids(words, oovs)
                batch_kg_extend_vocab[-1].append(kg_extend_ids)
                batch_kg_extend_len[-1].append(len(kg_extend_ids)) 
                batch_kg_ids[-1].append(ids)
                batch_kg_len[-1].append(len(ids))

            batch_oovs.append(oovs)
            batch_kg_nums.append(len(kg))

            batch_labels.append(self.vocab.response2ids(response_words + ['<eos>'], oovs))

        # history
        len_history = [len(history) for history in batch_history_ids]

        pad_history_ids = pad_data(batch_history_ids, pad_len=max(len_history), pad_idx=self.pad_idx)
        history_mask = len_to_mask(len_history)

        # response & labels
        len_response = [len(response) for response in batch_response_ids]
        pad_response_ids = pad_data(batch_response_ids, pad_len=max(len_response), pad_idx=self.pad_idx)
        pad_labels = pad_data(batch_labels, pad_len=max(len_response)-1, pad_idx=self.pad_idx)
        
        # knowledge
        pad_kg_len = max(max(item) for item in batch_kg_len)
        pad_kg_num = max(batch_kg_nums)
        pad_kg_ids = [pad_data(item, pad_len=pad_kg_len, pad_num=pad_kg_num, pad_idx=self.pad_idx) for item in batch_kg_ids]
        kg_mask = len_to_mask(batch_kg_nums)  # for knowledge selection

        # extend knowledge vocab
        pad_kg_extend_len = max(max(item) for item in batch_kg_extend_len)
        pad_kg_extend_vocab = [pad_data(item, pad_len=pad_kg_extend_len, pad_num=pad_kg_num, pad_idx=self.pad_idx) for item in batch_kg_extend_vocab]

        # pad kg_len
        pad_batch_kg_len = pad_data(batch_kg_len, pad_len=pad_kg_num, pad_idx=1)  # 用GRU/LSTM编码，pack的时候，sample对应的seq_len不能为0
        len_kg = []
        [len_kg.extend(item) for item in pad_batch_kg_len]

        # kg_padding)mask
        kg_padding_mask = len_to_mask(len_kg)
        # max oov words num
        max_kg_oov_num = max([len(item) for item in batch_oovs])
        
        return pad_history_ids, len_history, history_mask, \
                pad_response_ids, len_response, pad_labels, \
                pad_kg_ids, len_kg, kg_mask, pad_kg_len, \
                pad_kg_extend_vocab, kg_padding_mask, max_kg_oov_num, batch_oovs

        
    def __call__(self):
        return self.loader


def build_feed_data(batch_data, use_gpu=True, pre_train=False):
    pad_history_ids, len_history, history_mask, \
    pad_response_ids, len_response, pad_labels, \
    pad_kg_ids, len_kg, kg_mask, pad_kg_len, \
    pad_kg_extend_vocab, kg_padding_mask, max_kg_oov_num, _ = batch_data
    batch_size = len(len_history)
    if pre_train:
        kl_and_nll_factor = 0.
    else:
        kl_and_nll_factor = 1.
    feed_data = {
        'src_input': torch.tensor(pad_history_ids, dtype=torch.long),
        'src_len': torch.tensor(len_history, dtype=torch.float),
        'src_padding_mask': torch.tensor(history_mask, dtype=torch.float),

        'tgt_input': torch.tensor(pad_response_ids, dtype=torch.long),
        'tgt_len': torch.tensor(len_response, dtype=torch.long),
        'label': torch.tensor(pad_labels, dtype=torch.long),

        'kg_input': torch.tensor(pad_kg_ids, dtype=torch.long).reshape(-1, pad_kg_len),  # sample对应的knowledge聚合
        'kg_len': torch.tensor(len_kg, dtype=torch.long),
        'kg_mask': torch.tensor(kg_mask, dtype=torch.float),

        'kg_padding_mask': torch.tensor(kg_padding_mask),
        'kg_extend_vocab': torch.tensor(pad_kg_extend_vocab, dtype=torch.long),
        'extra_zero': torch.zeros(batch_size, max_kg_oov_num, dtype=torch.float),

        'kl_and_nll_factor': torch.tensor(kl_and_nll_factor, dtype=torch.float)
    }
    if use_gpu:
        for key in feed_data.keys():
            feed_data[key] = feed_data[key].cuda()

    return feed_data
