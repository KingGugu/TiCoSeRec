# -*- coding: utf-8 -*-

import copy
import random
import itertools
import numpy as np


def get_var(tlist):
    length = len(tlist)
    total = 0
    diffs = []

    if length == 1:
        return 0

    for i in range(length - 1):
        diff = abs(tlist[i + 1] - tlist[i])
        diffs.append(diff)
        total = total + diff
    avg_diff = total / len(diffs)

    total = 0
    for diff in diffs:
        total = total + (diff - avg_diff) ** 2
    result = total / len(diffs)

    return result


class CombinatorialEnumerate(object):
    """Given M type of augmentations, and a original sequence, successively call \
    the augmentation 2*C(M, 2) times can generate total C(M, 2) augmentaion pairs. 
    In another word, the augmentation method pattern will repeat after every 2*C(M, 2) calls.
    
    For example, M = 3, the argumentation methods to be called are in following order: 
    a1, a2, a1, a3, a2, a3. Which formed three pair-wise augmentations:
    (a1, a2), (a1, a3), (a2, a3) for multi-view contrastive learning.
    """

    def __init__(self, args, similarity_model):
        self.data_augmentation_methods = [Crop(args.crop_mode, args.crop_rate), Mask(args.mask_mode, args.mask_rate),
                                          Reorder(args.reorder_mode, args.reorder_rate),
                                          Insert(similarity_model, args.insert_mode, args.insert_rate,
                                                 args.max_insert_num_per_pos),
                                          Substitute(similarity_model, args.substitute_mode,
                                                     args.substitute_rate)]
        self.n_views = args.n_views
        self.augmentation_idx_list = self.__get_augmentation_idx_order()  # length of the list == C(M, 2)
        self.total_augmentation_samples = len(self.augmentation_idx_list)
        self.cur_augmentation_idx_of_idx = 0

    def __get_augmentation_idx_order(self):
        augmentation_idx_list = []
        for (view_1, view_2) in itertools.combinations([i for i in range(self.n_views)], 2):
            augmentation_idx_list.append(view_1)
            augmentation_idx_list.append(view_2)
        return augmentation_idx_list

    def __call__(self, item_sequence, time_sequence):
        augmentation_idx = self.augmentation_idx_list[self.cur_augmentation_idx_of_idx]
        augment_method = self.data_augmentation_methods[augmentation_idx]
        self.cur_augmentation_idx_of_idx += 1  # keep the index of index in range(0, C(M,2))
        self.cur_augmentation_idx_of_idx = self.cur_augmentation_idx_of_idx % self.total_augmentation_samples
        return augment_method(item_sequence, time_sequence)


class Random(object):
    """Randomly pick one data augmentation type every time call"""

    def __init__(self, args, similarity_model):
        self.short_seq_data_aug_methods = None
        self.augment_threshold = args.augment_threshold
        self.augment_type_for_short = args.augment_type_for_short
        if self.augment_threshold == -1:
            self.data_augmentation_methods = [Crop(args.crop_mode, args.crop_rate),
                                              Mask(args.mask_mode, args.mask_rate),
                                              Reorder(args.reorder_mode, args.reorder_rate),
                                              Insert(similarity_model, args.insert_mode, args.insert_rate,
                                                     args.max_insert_num_per_pos),
                                              Substitute(similarity_model, args.substitute_mode, args.substitute_rate)]
            print("Total augmentation numbers: ", len(self.data_augmentation_methods))
        elif self.augment_threshold > 0:
            print("short sequence augment type:", self.augment_type_for_short)
            self.short_seq_data_aug_methods = []
            if 'S' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(
                    Substitute(similarity_model, args.substitute_mode, args.substitute_rate))
            if 'I' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(
                    Insert(similarity_model, args.insert_mode, args.insert_rate, args.max_insert_num_per_pos))
            if 'M' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(Mask(args.mask_mode, args.mask_rate))
            if 'R' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(Reorder(args.reorder_mode, args.reorder_rate))
            if 'C' in self.augment_type_for_short:
                self.short_seq_data_aug_methods.append(Crop(args.crop_mode, args.crop_rate))
            if len(self.augment_type_for_short) == 5:
                print("all aug set for short sequences")
            self.long_seq_data_aug_methods = [Crop(args.crop_mode, args.crop_rate),
                                              Mask(args.mask_mode, args.mask_rate),
                                              Reorder(args.reorder_mode, args.reorder_rate),
                                              Insert(similarity_model, args.insert_mode, args.insert_rate,
                                                     args.max_insert_num_per_pos),
                                              Substitute(similarity_model, args.substitute_mode, args.substitute_rate)]
            print("Augmentation methods for Long sequences:", len(self.long_seq_data_aug_methods))
            print("Augmentation methods for short sequences:", len(self.short_seq_data_aug_methods))
        else:
            raise ValueError("Invalid data type.")

    def __call__(self, item_sequence, time_sequence):
        if self.augment_threshold == -1:
            # randint generate int x in range: a <= x <= b
            augment_method_idx = random.randint(0, len(self.data_augmentation_methods) - 1)
            augment_method = self.data_augmentation_methods[augment_method_idx]
            return augment_method(item_sequence, time_sequence)
        elif self.augment_threshold > 0:
            seq_len = len(item_sequence)
            if seq_len > self.augment_threshold:
                # randint generate int x in range: a <= x <= b
                augment_method_idx = random.randint(0, len(self.long_seq_data_aug_methods) - 1)
                augment_method = self.long_seq_data_aug_methods[augment_method_idx]
                return augment_method(item_sequence, time_sequence)
            elif seq_len <= self.augment_threshold:
                # randint generate int x in range: a <= x <= b
                augment_method_idx = random.randint(0, len(self.short_seq_data_aug_methods) - 1)
                augment_method = self.short_seq_data_aug_methods[augment_method_idx]
                return augment_method(item_sequence, time_sequence)


def _ensmeble_sim_models(top_k_one, top_k_two):
    # only support top k = 1 case so far
    if top_k_one[0][1] >= top_k_two[0][1]:
        return [top_k_one[0][0]]
    else:
        return [top_k_two[0][0]]


class Insert(object):
    """
    Insert similar items every time call.
    Priority is given to places with large time intervals.
    maximum: Insert at larger time intervals
    minimum: Insert at smaller time intervals
    """

    def __init__(self, item_similarity_model, mode, insert_rate=0.4, max_insert_num_per_pos=1):
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.mode = mode
        self.insert_rate = insert_rate
        self.max_insert_num_per_pos = max_insert_num_per_pos

    def __call__(self, item_sequence, time_sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(item_sequence)
        insert_nums = max(int(self.insert_rate * len(copied_sequence)), 1)

        time_diffs = []
        length = len(time_sequence)
        for i in range(length - 1):
            diff = abs(time_sequence[i + 1] - time_sequence[i])
            time_diffs.append(diff)
        assert self.mode in ['maximum', 'minimum']
        if self.mode == 'maximum':
            """
            First sort from large to small, and then return the original index value by sorting. 
            The larger the value, the smaller the index value
            """
            diff_sorted = np.argsort(time_diffs)[::-1]
        if self.mode == 'minimum':
            """
            First sort from small to large, and then return the original index value by sorting. 
            The larger the value, the larger the index.
            """
            diff_sorted = np.argsort(time_diffs)
        diff_sorted = diff_sorted.tolist()
        insert_idx = []
        for i in range(insert_nums):
            temp = diff_sorted[i]
            insert_idx.append(temp)

        """
        The index of time_diff is 1 smaller than the item. 
        The item should be inserted to the right of item_index. 
        Put the original item first in each cycle, so that the inserted item is inserted to the right of the original item
        """
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):

            inserted_sequence += [item]

            if index in insert_idx:
                top_k = random.randint(1, max(1, int(self.max_insert_num_per_pos / insert_nums)))
                if self.ensemble:
                    top_k_one = self.item_sim_model_1.most_similar(item, top_k=top_k, with_score=True)
                    top_k_two = self.item_sim_model_2.most_similar(item, top_k=top_k, with_score=True)
                    inserted_sequence += _ensmeble_sim_models(top_k_one, top_k_two)
                else:
                    inserted_sequence += self.item_similarity_model.most_similar(item, top_k=top_k)

        return inserted_sequence


class Substitute(object):
    """
    Substitute with similar items
    maximum: Substitute items with larger time interval
    minimum: Substitute items with smaller time interval
    """

    def __init__(self, item_similarity_model, mode, substitute_rate=0.1):
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.substitute_rate = substitute_rate
        self.mode = mode

    def __call__(self, item_sequence, time_sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(item_sequence)
        if len(copied_sequence) <= 1:
            return copied_sequence
        substitute_nums = max(int(self.substitute_rate * len(copied_sequence)), 1)

        time_diffs = []
        length = len(time_sequence)
        for i in range(length - 1):
            diff = abs(time_sequence[i + 1] - time_sequence[i])
            time_diffs.append(diff)

        diff_sorted = []
        assert self.mode in ['maximum', 'minimum']
        if self.mode == 'maximum':
            """
            First sort from large to small, and then return the original index value by sorting. 
            The larger the value, the smaller the index value
            """
            diff_sorted = np.argsort(time_diffs)[::-1]
        if self.mode == 'minimum':
            """
            First sort from small to large, and then return the original index value by sorting. 
            The larger the value, the larger the index.
            """
            diff_sorted = np.argsort(time_diffs)
        diff_sorted = diff_sorted.tolist()
        substitute_idx = []
        for i in range(substitute_nums):
            temp = diff_sorted[i]
            substitute_idx.append(temp)

        for index in substitute_idx:
            if self.ensemble:
                top_k_one = self.item_sim_model_1.most_similar(copied_sequence[index], with_score=True)
                top_k_two = self.item_sim_model_2.most_similar(copied_sequence[index], with_score=True)
                substitute_items = _ensmeble_sim_models(top_k_one, top_k_two)
                copied_sequence[index] = substitute_items[0]
            else:
                copied_sequence[index] = copied_sequence[index] = \
                    self.item_similarity_model.most_similar(copied_sequence[index])[0]
        return copied_sequence


class Crop(object):
    """
    maximum: Crop subsequences with the maximum time interval variance
    minimum: Crop subsequences with the minimum time interval variance
    """

    def __init__(self, mode, tao=0.2):
        self.tao = tao
        self.mode = mode

    def __call__(self, item_sequence, time_sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(item_sequence)
        sub_seq_length = int(self.tao * len(copied_sequence))
        # randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        if sub_seq_length <= 2:
            return [copied_sequence[start_index]]

        cropped_vars = []
        crop_index = []
        for i in range(len(item_sequence)):
            if len(item_sequence) - i - sub_seq_length >= 0:
                left_index = len(item_sequence) - i - sub_seq_length
                right_index = left_index + sub_seq_length
                temp_time_sequence = time_sequence[left_index:right_index - 1]
                temp_var = get_var(temp_time_sequence)

                cropped_vars.append(temp_var)
                crop_index.append(left_index)
        temp = []
        assert self.mode in ['maximum', 'minimum']
        if self.mode == 'maximum':
            temp = cropped_vars.index(max(cropped_vars))
        if self.mode == 'minimum':
            temp = cropped_vars.index(min(cropped_vars))
        start_index = crop_index.index(temp)

        cropped_sequence = copied_sequence[start_index:start_index + sub_seq_length]
        return cropped_sequence


class Mask(object):
    """
    Randomly mask k items given a sequence
    maximum: Mask items with larger time interval
    minimum: Mask items with smaller time interval
    """

    def __init__(self, mode, gamma=0.7):
        self.gamma = gamma
        self.mode = mode

    def __call__(self, item_sequence, time_sequence):
        copied_sequence = copy.deepcopy(item_sequence)
        mask_nums = int(self.gamma * len(copied_sequence))
        mask = [0 for i in range(mask_nums)]

        if len(copied_sequence) <= 1:
            return copied_sequence

        time_diffs = []
        length = len(time_sequence)
        for i in range(length - 1):
            diff = abs(time_sequence[i + 1] - time_sequence[i])
            time_diffs.append(diff)

        diff_sorted = []
        assert self.mode in ['maximum', 'minimum', 'random']
        if self.mode == 'random':
            copied_sequence = copy.deepcopy(item_sequence)
            mask_nums = int(self.gamma * len(copied_sequence))
            mask = [0 for i in range(mask_nums)]
            mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
            for idx, mask_value in zip(mask_idx, mask):
                copied_sequence[idx] = mask_value
            return copied_sequence
        if self.mode == 'maximum':
            """
            First sort from large to small, and then return the original index value by sorting. 
            The larger the value, the smaller the index value
            """
            diff_sorted = np.argsort(time_diffs)[::-1]
        if self.mode == 'minimum':
            """
            First sort from small to large, and then return the original index value by sorting. 
            The larger the value, the larger the index.
            """
            diff_sorted = np.argsort(time_diffs)
        diff_sorted = diff_sorted.tolist()
        mask_idx = []
        for i in range(mask_nums):
            temp = diff_sorted[i]
            mask_idx.append(temp)

        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence


class Reorder(object):
    """
    Randomly shuffle a continuous sub-sequence
    maximum: Reorder subsequences with the maximum time interval variance
    minimum: Reorder subsequences with the minimum variance of time interval
    """

    def __init__(self, mode, beta=0.2):
        self.beta = beta
        self.mode = mode

    def __call__(self, item_sequence, time_sequence):
        copied_sequence = copy.deepcopy(item_sequence)
        sub_seq_length = int(self.beta * len(copied_sequence))
        if sub_seq_length < 2:
            return copied_sequence

        cropped_vars = []
        crop_index = []
        for i in range(len(item_sequence)):
            if len(item_sequence) - i - sub_seq_length >= 0:
                left_index = len(item_sequence) - i - sub_seq_length
                right_index = left_index + sub_seq_length
                temp_time_sequence = time_sequence[left_index:right_index - 1]
                temp_var = get_var(temp_time_sequence)

                cropped_vars.append(temp_var)
                crop_index.append(left_index)
        temp = []
        assert self.mode in ['maximum', 'minimum']
        if self.mode == 'maximum':
            temp = cropped_vars.index(max(cropped_vars))
        if self.mode == 'minimum':
            temp = cropped_vars.index(min(cropped_vars))
        start_index = crop_index.index(temp)

        sub_seq = copied_sequence[start_index:start_index + sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + copied_sequence[start_index + sub_seq_length:]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq
