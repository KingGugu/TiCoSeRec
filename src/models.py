# -*- coding:utf-8 -*-

import os
import math
import copy
import pickle
import gensim
import random

import torch
import torch.nn as nn

from tqdm import tqdm
from modules import TransformerEncoder, get_attention_mask


class SASRec(nn.Module):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, args):
        super(SASRec, self).__init__()

        # load parameters info
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.hidden_size = args.hidden_size  # same as embedding_size
        self.inner_size = args.inner_size  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = args.hidden_dropout_prob
        self.attn_dropout_prob = args.attn_dropout_prob
        self.hidden_act = args.hidden_act
        self.layer_norm_eps = args.layer_norm_eps
        self.item_size = args.item_size
        self.max_seq_length = args.max_seq_length
        self.args = args

        self.initializer_range = args.initializer_range

        # define layers and loss
        self.item_embedding = nn.Embedding(self.item_size, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transformer_encoder(self, item_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        return output


class OnlineItemSimilarity:

    def __init__(self, item_size):
        self.item_size = item_size
        self.item_embedding = None
        self.cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.total_item_list = torch.tensor([i for i in range(self.item_size)],
                                            dtype=torch.long).to(self.device)

    def update_embedding_matrix(self, item_embedding):
        self.item_embedding = copy.deepcopy(item_embedding)
        self.base_embedding_matrix = self.item_embedding(self.total_item_list)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item_idx in range(1, self.item_size):
            try:
                item_vector = self.item_embedding(torch.tensor(item_idx).to(self.device)).view(-1, 1)
                item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
                max_score = max(torch.max(item_similarity), max_score)
                min_score = min(torch.min(item_similarity), min_score)
            except:
                continue
        return max_score, min_score

    def most_similar(self, item_idx, top_k=1, with_score=False):
        item_idx = torch.tensor(item_idx, dtype=torch.long).to(self.device)
        item_vector = self.item_embedding(item_idx).view(-1, 1)
        item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
        item_similarity = (self.max_score - item_similarity) / (self.max_score - self.min_score)
        # remove item idx itself
        values, indices = item_similarity.topk(top_k + 1)
        if with_score:
            item_list = indices.tolist()
            score_list = values.tolist()
            if item_idx in item_list:
                idd = item_list.index(item_idx)
                item_list.remove(item_idx)
                score_list.pop(idd)
            return list(zip(item_list, score_list))
        item_list = indices.tolist()
        if item_idx in item_list:
            item_list.remove(item_idx)
        return item_list


class OfflineItemSimilarity:
    def __init__(self, data_file=None, similarity_path=None, model_name='ItemCF', dataset_name='Sports_and_Outdoors'):
        self.dataset_name = dataset_name
        self.similarity_path = similarity_path
        # train_data_list used for item2vec, train_data_dict used for itemCF and itemCF-IUF
        self.train_data_list, self.train_item_list, self.train_data_dict = self._load_train_data(data_file)
        self.model_name = model_name
        self.similarity_model = self.load_similarity_model(self.similarity_path)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item in self.similarity_model.keys():
            for neig in self.similarity_model[item]:
                sim_score = self.similarity_model[item][neig]
                max_score = max(max_score, sim_score)
                min_score = min(min_score, sim_score)
        return max_score, min_score

    def _convert_data_to_dict(self, data):
        """
        split the data set
        testdata is a test data set
        traindata is a train set
        """
        train_data_dict = {}
        for user, item, record in data:
            train_data_dict.setdefault(user, {})
            train_data_dict[user][item] = record
        return train_data_dict

    def _save_dict(self, dict_data, save_path='./similarity.pkl'):
        print("saving data to ", save_path)
        with open(save_path, 'wb') as write_file:
            pickle.dump(dict_data, write_file)

    def _load_train_data(self, data_file=None):
        """
        read the data from the data file which is a data set
        """
        train_data = []
        train_data_list = []
        train_data_set_list = []
        for line in open(data_file).readlines():
            userid, items = line.strip().split(' ', 1)
            # only use training data
            items = items.split(' ')[:-3]
            train_data_list.append(items)
            train_data_set_list += items
            for itemid in items:
                train_data.append((userid, itemid, int(1)))
        return train_data_list, set(train_data_set_list), self._convert_data_to_dict(train_data)

    def _generate_item_similarity(self, train=None, save_path='./'):
        """
        calculate co-rated users between items
        """
        print("getting item similarity...")
        train = train or self.train_data_dict
        C = dict()
        N = dict()

        if self.model_name in ['ItemCF', 'ItemCF_IUF']:
            print("Step 1: Compute Statistics")
            data_iter = tqdm(enumerate(train.items()), total=len(train.items()))
            for idx, (u, items) in data_iter:
                if self.model_name == 'ItemCF':
                    for i in items.keys():
                        N.setdefault(i, 0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i, {})
                            C[i].setdefault(j, 0)
                            C[i][j] += 1
                elif self.model_name == 'ItemCF_IUF':
                    for i in items.keys():
                        N.setdefault(i, 0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i, {})
                            C[i].setdefault(j, 0)
                            C[i][j] += 1 / math.log(1 + len(items) * 1.0)
            self.itemSimBest = dict()
            print("Step 2: Compute co-rate matrix")
            c_iter = tqdm(enumerate(C.items()), total=len(C.items()))
            for idx, (cur_item, related_items) in c_iter:
                self.itemSimBest.setdefault(cur_item, {})
                for related_item, score in related_items.items():
                    self.itemSimBest[cur_item].setdefault(related_item, 0)
                    self.itemSimBest[cur_item][related_item] = score / math.sqrt(N[cur_item] * N[related_item])
            self._save_dict(self.itemSimBest, save_path=save_path)
        elif self.model_name == 'Item2Vec':
            # details here: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py
            print("Step 1: train item2vec model")
            item2vec_model = gensim.models.Word2Vec(sentences=self.train_data_list,
                                                    vector_size=20, window=5, min_count=0,
                                                    epochs=100)
            self.itemSimBest = dict()
            total_item_nums = len(item2vec_model.wv.index_to_key)
            print("Step 2: convert to item similarity dict")
            total_items = tqdm(item2vec_model.wv.index_to_key, total=total_item_nums)
            for cur_item in total_items:
                related_items = item2vec_model.wv.most_similar(positive=[cur_item], topn=20)
                self.itemSimBest.setdefault(cur_item, {})
                for (related_item, score) in related_items:
                    self.itemSimBest[cur_item].setdefault(related_item, 0)
                    self.itemSimBest[cur_item][related_item] = score
            print("Item2Vec model saved to: ", save_path)
            self._save_dict(self.itemSimBest, save_path=save_path)

    def load_similarity_model(self, similarity_model_path):
        if not similarity_model_path:
            raise ValueError('invalid path')
        elif not os.path.exists(similarity_model_path):
            print("the similirity dict not exist, generating...")
            self._generate_item_similarity(save_path=self.similarity_path)
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'LightGCN']:
            with open(similarity_model_path, 'rb') as read_file:
                similarity_dict = pickle.load(read_file)
            return similarity_dict
        elif self.model_name == 'Random':
            similarity_dict = self.train_item_list
            return similarity_dict

    def most_similar(self, item, top_k=1, with_score=False):
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'LightGCN']:
            """TODO: handle case that item not in keys"""
            if str(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[str(item)].items(), key=lambda x: x[1],
                                                reverse=True)[0:top_k]
                if with_score:
                    return list(
                        map(lambda x: (int(x[0]), (self.max_score - float(x[1])) / (self.max_score - self.min_score)),
                            top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            elif int(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[int(item)].items(), key=lambda x: x[1],
                                                reverse=True)[0:top_k]
                if with_score:
                    return list(
                        map(lambda x: (int(x[0]), (self.max_score - float(x[1])) / (self.max_score - self.min_score)),
                            top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            else:
                item_list = list(self.similarity_model.keys())
                random_items = random.sample(item_list, k=top_k)
                if with_score:
                    return list(map(lambda x: (int(x), 0.0), random_items))
                return list(map(lambda x: int(x), random_items))
        elif self.model_name == 'Random':
            random_items = random.sample(self.similarity_model, k=top_k)
            if with_score:
                return list(map(lambda x: (int(x), 0.0), random_items))
            return list(map(lambda x: int(x), random_items))
