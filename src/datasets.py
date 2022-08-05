# -*- coding: utf-8 -*-

import copy
import torch
import random

from torch.utils.data import Dataset

from utils import neg_sample, nCr
from data_augmentation_time import Crop, Mask, Reorder, Substitute, Insert, Random, CombinatorialEnumerate


class RecWithContrastiveLearningDataset(Dataset):
    def __init__(self, args, user_seq, time_seq, not_aug_users=None, data_type='train',
                 similarity_model_type='offline', total_train_users=0):
        self.args = args
        self.user_seq = user_seq
        self.time_seq = time_seq
        self.data_type = data_type
        self.max_len = args.max_seq_length
        self.not_aug_users = not_aug_users

        self.total_train_users = total_train_users
        self.model_warm_up_train_users = args.model_warm_up_epochs * len(user_seq)

        # currently apply one transform, will extend to multiples
        # it takes one sequence of items as input, and apply augmentation operation to get another sequence
        if similarity_model_type == 'offline':
            self.similarity_model = args.offline_similarity_model
        elif similarity_model_type == 'online':
            self.similarity_model = args.online_similarity_model
        elif similarity_model_type == 'hybrid':
            self.similarity_model = [args.offline_similarity_model, args.online_similarity_model]
        print("Similarity Model Type:", similarity_model_type)
        self.augmentations = {'crop': Crop(args.crop_mode, args.crop_rate),
                              'mask': Mask(args.mask_mode, args.mask_rate),
                              'reorder': Reorder(args.reorder_mode, args.reorder_rate),
                              'substitute': Substitute(self.similarity_model, args.substitute_mode,
                                                       args.substitute_rate),
                              'insert': Insert(self.similarity_model, args.insert_rate, args.max_insert_num_per_pos),
                              'random': Random(args, self.similarity_model),
                              'combinatorial_enumerate': CombinatorialEnumerate(args, self.similarity_model)}

        if self.args.base_augment_type not in self.augmentations:
            raise ValueError(f"augmentation type: '{self.args.base_augment_type}' is invalided")
        print(f"Creating Contrastive Learning Dataset using '{self.args.base_augment_type}' data augmentation")
        self.base_transform = self.augmentations[self.args.base_augment_type]
        # number of augmentations for each sequences, current support two
        self.n_views = self.args.n_views

    def _one_pair_data_augmentation(self, input_ids, input_times, not_aug=False):
        """
        provides two positive samples given one sequence
        """
        augmented_seqs = []
        for i in range(2):
            if not_aug:
                assert self.args.not_aug_data_mode in ['zero', 'original']
                if self.args.not_aug_data_mode == 'zero':
                    # Return not augmented user data with 0
                    augmented_input_ids = [0] * self.max_len

                if self.args.not_aug_data_mode == 'original':
                    # Return not augmented user data with original data
                    pad_len = self.max_len - len(input_ids)
                    augmented_input_ids = [0] * pad_len + input_ids
                    augmented_input_ids = augmented_input_ids[-self.max_len:]
            else:
                augmented_input_ids = self.base_transform(input_ids, input_times)
                pad_len = self.max_len - len(augmented_input_ids)
                augmented_input_ids = [0] * pad_len + augmented_input_ids
                augmented_input_ids = augmented_input_ids[-self.max_len:]

            assert len(augmented_input_ids) == self.max_len
            cur_tensors = (torch.tensor(augmented_input_ids, dtype=torch.long))
            augmented_seqs.append(cur_tensors)
        return augmented_seqs

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in copied_input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(copied_input_ids)
        copied_input_ids = [0] * pad_len + copied_input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        copied_input_ids = copied_input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(copied_input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(copied_input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_rec_tensors

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]
        times = self.time_seq[index]
        input_times = times[:-2]

        self.total_train_users += 1

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use
        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]
        else:
            items_with_noise = self._add_noise_interactions(items)
            input_ids = items_with_noise[:-1]
            target_pos = items_with_noise[1:]
            answer = [items_with_noise[-1]]

        if self.data_type == "train":
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            cf_tensors_list = []
            not_aug = False
            # if n_views == 2, then it's downgraded to pair-wise contrastive learning
            total_augmentation_pairs = nCr(self.n_views, 2)
            if self.total_train_users <= self.model_warm_up_train_users:
                total_augmentation_pairs = 0
            if (user_id in self.not_aug_users) and (self.total_train_users > self.model_warm_up_train_users):
                not_aug = True
            for i in range(total_augmentation_pairs):
                cf_tensors_list.append(self._one_pair_data_augmentation(input_ids, input_times, not_aug))
            return cur_rec_tensors, cf_tensors_list
        elif self.data_type == 'valid':
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            return cur_rec_tensors
        else:
            cur_rec_tensors = self._data_sample_rec_task(user_id, items_with_noise, input_ids, target_pos, answer)
            return cur_rec_tensors

    def __len__(self):
        """
        consider n_view of a single sequence as one sample
        """
        return len(self.user_seq)
