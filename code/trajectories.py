import os
import math

import numpy as np
from torch.utils.data import DataLoader

import torch
from torch.utils.data import Dataset
from constants import *


def data_loader(path, metric, train_or_test):
    dset = Preprocessor(path, metric, train_or_test)

    if MULTI_CONDITIONAL_MODEL:
        loader = DataLoader(dset, batch_size=BATCH_MULTI_CONDITION, shuffle=True, num_workers=NUM_WORKERS, collate_fn=seq_collate)
    else:
        loader = DataLoader(dset, batch_size=BATCH_SINGLE_CONDITION, shuffle=True, num_workers=NUM_WORKERS, collate_fn=seq_collate)
    return dset, loader


def seq_collate(data):
    if MULTI_CONDITIONAL_MODEL:
        (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, loss_mask_list, obs_obj_abs_speed,
        pred_obj_abs_speed, obs_label, pred_label, obs_obj_rel_speed) = zip(*data)
    else:
        (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, loss_mask_list, obs_obj_abs_speed,
         pred_obj_abs_speed, obs_obj_rel_speed) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    obs_obj_abs_speed = torch.cat(obs_obj_abs_speed, dim=0).permute(2, 0, 1)
    pred_obj_abs_speed = torch.cat(pred_obj_abs_speed, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    obs_obj_rel_speed = torch.cat(obs_obj_rel_speed, dim=0).permute(2, 0, 1)

    if MULTI_CONDITIONAL_MODEL:
        obs_label = torch.cat(obs_label, dim=0).permute(2, 0, 1)
        pred_label = torch.cat(pred_label, dim=0).permute(2, 0, 1)

    if MULTI_CONDITIONAL_MODEL:
        out = [
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, loss_mask, seq_start_end, obs_obj_abs_speed,
            pred_obj_abs_speed, obs_label, pred_label, obs_obj_rel_speed
        ]
    else:
        out = [
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, loss_mask, seq_start_end, obs_obj_abs_speed,
            pred_obj_abs_speed, obs_obj_rel_speed
        ]

    return tuple(out)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def read_file(_path):
    data = []
    i = 0
    if MULTI_CONDITIONAL_MODEL:
        with open(_path, 'r') as f:
            for line in f:
                if i == 0:
                    i += 1
                    continue
                line = line.strip().split(',')
                line = [i for i in line]
                data.append(line)
    else:
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                line = [float(i) for i in line]
                data.append(line)
    return np.asarray(data)


class Preprocessor(Dataset):

    def __init__(
            self, data_dir, metric=0, train_or_test = None
    ):
        super(Preprocessor, self).__init__()

        self.data_dir = data_dir
        SEQ_LEN = OBS_LEN + PRED_LEN
        self.train_or_test = metric
        self.train_or_test = train_or_test

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_obj_in_seq = []
        seq_list = []
        seq_list_rel = []
        obj_abs_speed = []
        obj_rel_speed = []
        obj_label = []
        loss_mask_list = []
        for path in all_files:
            data = read_file(path)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - SEQ_LEN + 1)))

            for idx in range(0, num_sequences + 1):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + SEQ_LEN], axis=0)

                obj_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_loss_mask = np.zeros((len(obj_in_curr_seq), SEQ_LEN))
                curr_seq_rel = np.zeros((len(obj_in_curr_seq), 2, SEQ_LEN))
                curr_seq_rel_speed = np.zeros((len(obj_in_curr_seq), SEQ_LEN))
                curr_seq = np.zeros((len(obj_in_curr_seq), 2, SEQ_LEN))
                _curr_obj_abs_speed = np.zeros((len(obj_in_curr_seq), SEQ_LEN))
                _curr_obj_label = np.zeros((len(obj_in_curr_seq), 3, SEQ_LEN))
                num_obj_considered = 0

                for _, obj_id in enumerate(obj_in_curr_seq):
                    curr_obj_seq = curr_seq_data[curr_seq_data[:, 1] == obj_id, :]
                    if MULTI_CONDITIONAL_MODEL:
                        label = curr_obj_seq[0, 2]
                    pad_front = frames.index(curr_obj_seq[0, 0]) - idx
                    pad_end = frames.index(curr_obj_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != SEQ_LEN:
                        continue
                    if MULTI_CONDITIONAL_MODEL:
                        if len(curr_obj_seq[:, 0]) != SEQ_LEN:
                            continue
                    if MULTI_CONDITIONAL_MODEL:
                        curr_obj_x_axis_new = [0.0] + [np.square(float(t) - float(s)) for s, t in
                                                   zip(curr_obj_seq[:, 3], curr_obj_seq[1:, 3])]
                        curr_obj_y_axis_new = [0.0] + [np.square(float(t) - float(s)) for s, t in
                                                   zip(curr_obj_seq[:, 4], curr_obj_seq[1:, 4])]
                        curr_obj_dist = np.sqrt(np.add(curr_obj_x_axis_new, curr_obj_y_axis_new))
                    else:
                        if EXTRAPOLATE_MAX or EXTRAPOLATE_MID or EXTRAPOLATE_MIN:
                            curr_obj_x_axis_new = [np.square(t - s) for s, t in
                                                           zip(curr_obj_seq[:, 2], curr_obj_seq[1:, 2])]
                            curr_obj_y_axis_new = [np.square(t - s) for s, t in
                                                           zip(curr_obj_seq[:, 3], curr_obj_seq[1:, 3])]
                            curr_obj_dist = np.sqrt(np.add(curr_obj_x_axis_new, curr_obj_y_axis_new))
                            curr_obj_dist = np.insert(curr_obj_dist, 0, curr_obj_dist[0])
                        else:
                            curr_obj_x_axis_new = [0.0] + [np.square(t - s) for s, t in
                                                           zip(curr_obj_seq[:, 2], curr_obj_seq[1:, 2])]
                            curr_obj_y_axis_new = [0.0] + [np.square(t - s) for s, t in
                                                           zip(curr_obj_seq[:, 3], curr_obj_seq[1:, 3])]

                            curr_obj_dist = np.sqrt(np.add(curr_obj_x_axis_new, curr_obj_y_axis_new))
                    if SINGLE_CONDITIONAL_MODEL:
                        curr_obj_abs_speed = curr_obj_dist / FRAMES_PER_SECOND_SINGLE_CONDITION
                    else:
                        curr_obj_abs_speed = curr_obj_dist / (FRAMES_PER_SECOND_MULTI_CONDITION * NORMALIZATION_FACTOR)
                    if EXTRAPOLATE_MIN:
                        if train_or_test == 'train' or train_or_test == 'val':
                            if any(0.0 <= i <= 0.66 for i in curr_obj_abs_speed):
                                continue
                        else:
                            if any(i > 0.66 for i in curr_obj_abs_speed):
                                continue
                    elif EXTRAPOLATE_MID:
                        if train_or_test == 'train' or train_or_test == 'val':
                            if any(0.66 <= i <= 1.32 for i in curr_obj_abs_speed):
                                continue
                        else:
                            if any(0.66 > i > 1.32 for i in curr_obj_abs_speed):
                                continue
                    elif EXTRAPOLATE_MAX:
                        if train_or_test == 'train' or train_or_test == 'val':
                            if any(i >= 1.32 for i in curr_obj_abs_speed):
                                continue
                        else:
                            if any(i < 1.32 for i in curr_obj_abs_speed):
                                continue

                    curr_obj_abs_speed = [sigmoid(x) for x in curr_obj_abs_speed]
                    curr_obj_abs_speed = np.around(curr_obj_abs_speed, decimals=4)
                    curr_obj_abs_speed = np.transpose(curr_obj_abs_speed)
                    _idx = num_obj_considered

                    if MULTI_CONDITIONAL_MODEL:
                        # ONE HOT ENCODING OF AGENT LABELS
                        emb_label = np.zeros(3, 'uint8')
                        if label == 'AV':
                            emb_label[0] = 1
                        elif label == 'OTHERS':
                            emb_label[1] = 1
                        elif label == 'AGENT':
                            emb_label[2] = 1
                        curr_obj_seq = np.transpose(curr_obj_seq[:, 3:5])
                        emb_label = np.repeat(emb_label, SEQ_LEN).reshape(3, -1)
                        _curr_obj_label[_idx, :, pad_front:pad_end] = emb_label

                    else:
                        curr_obj_seq = np.transpose(curr_obj_seq[:, 2:])
                    curr_obj_seq = curr_obj_seq.astype(float)
                    curr_obj_seq = np.around(curr_obj_seq, decimals=4)

                    rel_curr_obj_seq = np.zeros(curr_obj_seq.shape)
                    rel_curr_obj_seq[:, 1:] = curr_obj_seq[:, 1:] - curr_obj_seq[:, :-1]
                    curr_seq[_idx, :, pad_front:pad_end] = curr_obj_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_obj_seq

                    rel_curr_obj_speed = np.zeros(curr_obj_abs_speed.shape)
                    rel_curr_obj_speed[1:] = curr_obj_abs_speed[1:] - curr_obj_abs_speed[:-1]

                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    _curr_obj_abs_speed[_idx, pad_front:pad_end] = curr_obj_abs_speed
                    curr_seq_rel_speed[_idx, pad_front:pad_end] = rel_curr_obj_speed
                    num_obj_considered += 1

                if num_obj_considered > 1:
                    num_obj_in_seq.append(num_obj_considered)
                    loss_mask_list.append(curr_loss_mask[:num_obj_considered])
                    obj_abs_speed.append(_curr_obj_abs_speed[:num_obj_considered])
                    if MULTI_CONDITIONAL_MODEL:
                        obj_label.append(_curr_obj_label[:num_obj_considered])
                    seq_list.append(curr_seq[:num_obj_considered])
                    seq_list_rel.append(curr_seq_rel[:num_obj_considered])
                    obj_rel_speed.append(curr_seq_rel_speed[:num_obj_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        obj_abs_speed = np.concatenate(obj_abs_speed, axis=0)
        obj_rel_speed = np.concatenate(obj_rel_speed, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        obj_abs_speed = torch.from_numpy(obj_abs_speed).type(torch.float)
        obj_rel_speed = torch.from_numpy(obj_rel_speed).type(torch.float)
        if MULTI_CONDITIONAL_MODEL:
            obj_label = np.concatenate(obj_label, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, :OBS_LEN]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, OBS_LEN:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :OBS_LEN]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, OBS_LEN:]).type(torch.float)

        self.obs_obj_abs_speed = obj_abs_speed[:, :OBS_LEN].unsqueeze(dim=1).type(torch.float)
        self.pred_obj_abs_speed = obj_abs_speed[:, OBS_LEN:].unsqueeze(dim=1).type(torch.float)

        self.obs_obj_rel_speed = obj_rel_speed[:, :OBS_LEN].unsqueeze(dim=1).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)

        if MULTI_CONDITIONAL_MODEL:
            self.obs_obj_label = torch.from_numpy(obj_label[:, :, :OBS_LEN]).type(torch.float)
            self.pred_obj_label = torch.from_numpy(obj_label[:, :, OBS_LEN:]).type(torch.float)

        cum_start_idx = [0] + np.cumsum(num_obj_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        if MULTI_CONDITIONAL_MODEL:
            out = [
                self.obs_traj[start:end, :], self.pred_traj[start:end, :],
                self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
                self.loss_mask[start:end, :], self.obs_obj_abs_speed[start:end, :],
                self.pred_obj_abs_speed[start:end, :], self.obs_obj_label[start:end, :],
                self.pred_obj_label[start:end, :], self.obs_obj_rel_speed[start:end, :]
            ]
        else:
            out = [
                self.obs_traj[start:end, :], self.pred_traj[start:end, :],
                self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
                self.loss_mask[start:end, :], self.obs_obj_abs_speed[start:end, :],
                self.pred_obj_abs_speed[start:end, :], self.obs_obj_rel_speed[start:end, :]
            ]
        return out