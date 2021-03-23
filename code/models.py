import torch
import torch.nn as nn
from constants import *
import math
from scipy.spatial.distance import pdist, squareform
import numpy as np
import torch.nn.functional as F


def make_mlp(dim_list, activation='leakyrelu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class SpeedEncoderDecoder(nn.Module):
    """Speed Regressor Component to predict agent's speed"""
    def __init__(self, h_dim):
        super(SpeedEncoderDecoder, self).__init__()

        self.embedding_dim = EMBEDDING_DIM
        self.num_layers = NUM_LAYERS
        self.h_dim = h_dim

        self.speed_decoder = nn.LSTM(EMBEDDING_DIM, h_dim, NUM_LAYERS, dropout=DROPOUT)
        self.speed_mlp = nn.Linear(h_dim, 1)
        self.speed_embedding = nn.Linear(1, EMBEDDING_DIM)

    def init_hidden(self, batch):
        if USE_GPU:
            c_s, r_s = torch.zeros(self.num_layers, batch, self.h_dim).cuda(), torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        else:
            c_s, r_s = torch.zeros(self.num_layers, batch, self.h_dim), torch.zeros(self.num_layers, batch, self.h_dim)
        return c_s, r_s

    def forward(self, obs_speed, final_enc_h, label=None):
        sig_layer = nn.Sigmoid()
        batch = obs_speed.size(1)
        pred_speed_fake = []
        final_enc_h = final_enc_h.view(-1, self.h_dim)
        next_speed = obs_speed[-1, :, :]
        decoder_input = self.speed_embedding(next_speed.view(-1, 1))
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        decoder_h = final_enc_h.unsqueeze(dim=0)  # INITIALIZE THE DECODER HIDDEN STATE
        if USE_GPU:
            decoder_c = torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        else:
            decoder_c = torch.zeros(self.num_layers, batch, self.h_dim)

        state_tuple = (decoder_h, decoder_c)  # INITIALIZE THE STATE TUPLES

        for id in range(PRED_LEN):
            output, state_tuple = self.speed_decoder(decoder_input, state_tuple)
            next_dec_speed = self.speed_mlp(output.view(-1, self.h_dim))
            next_speed = sig_layer(next_dec_speed.view(-1, 1))
            decoder_input = self.speed_embedding(next_speed.view(-1, 1))
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)

            pred_speed_fake.append(next_speed.view(-1, 1))

        pred_speed_fake = torch.stack(pred_speed_fake, dim=0)
        return pred_speed_fake


class Encoder(nn.Module):
    def __init__(self, h_dim, mlp_input_dim):
        super(Encoder, self).__init__()

        self.mlp_dim = MLP_DIM
        self.h_dim = h_dim
        self.embedding_dim = EMBEDDING_DIM
        self.num_layers = NUM_LAYERS
        self.mlp_input_dim = mlp_input_dim

        self.encoder = nn.LSTM(EMBEDDING_DIM, h_dim, NUM_LAYERS, dropout=DROPOUT)

        self.spatial_embedding = nn.Linear(mlp_input_dim, EMBEDDING_DIM)

    def init_hidden(self, batch):
        if USE_GPU:
            c_s, r_s = torch.zeros(self.num_layers, batch, self.h_dim).cuda(), torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        else:
            c_s, r_s = torch.zeros(self.num_layers, batch, self.h_dim), torch.zeros(self.num_layers, batch, self.h_dim)
        return c_s, r_s

    def forward(self, obs_traj, obs_ped_speed, label=None):
        batch = obs_traj.size(1)
        if MULTI_CONDITIONAL_MODEL:
            embedding_input = torch.cat([obs_traj, obs_ped_speed, label], dim=2)
        else:
            embedding_input = torch.cat([obs_traj, obs_ped_speed], dim=2)
        traj_speed_embedding = self.spatial_embedding(embedding_input.contiguous().view(-1, self.mlp_input_dim))
        obs_traj_embedding = traj_speed_embedding.view(-1, batch, self.embedding_dim)
        # Initializing Encoder hidden states with zeroes
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Decoder(nn.Module):
    def __init__(self, h_dim, mlp_input_dim):
        super(Decoder, self).__init__()

        self.mlp_dim = MLP_DIM
        self.h_dim = h_dim
        self.embedding_dim = EMBEDDING_DIM
        self.mlp_input_dim = mlp_input_dim

        self.decoder = nn.LSTM(EMBEDDING_DIM, h_dim, NUM_LAYERS, dropout=DROPOUT)

        self.spatial_embedding = nn.Linear(mlp_input_dim, EMBEDDING_DIM)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end, pred_ped_speed, train_or_test, fake_ped_speed, label=None):
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        if train_or_test == 0:
            if MULTI_CONDITIONAL_MODEL:
                last_pos_speed = torch.cat([last_pos_rel, pred_ped_speed[0, :, :], label[0, :, :]], dim=1)
            else:
                last_pos_speed = torch.cat([last_pos_rel, pred_ped_speed[0, :, :]], dim=1)
        elif train_or_test == 1:  # USED FOR PREDICTION PURPOSE
            if MULTI_CONDITIONAL_MODEL:
                last_pos_speed = torch.cat([last_pos_rel, fake_ped_speed[0, :, :], label[0, :, :]], dim=1)
            else:
                last_pos_speed = torch.cat([last_pos_rel, fake_ped_speed[0, :, :]], dim=1)
        else:  # USED FOR SIMULATION PURPOSE
            if MULTI_CONDITIONAL_MODEL:
                next_speed = speed_control(pred_ped_speed[0, :, :], seq_start_end, label=label[0, :, :])
                last_pos_speed = torch.cat([last_pos_rel, next_speed, label[0, :, :]], dim=1)
            else:
                next_speed = speed_control(pred_ped_speed[0, :, :], seq_start_end)
                last_pos_speed = torch.cat([last_pos_rel, next_speed], dim=1)
        decoder_input = self.spatial_embedding(last_pos_speed)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for id in range(PRED_LEN):
            # At first prediction timestep, we initialize the decoder with Encoder hidden states + aggregation component
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos
            if id + 1 != PRED_LEN:
                if train_or_test == 0:  # GT used during CSG training
                    speed = pred_ped_speed[id + 1, :, :]
                    if MULTI_CONDITIONAL_MODEL:
                        curr_label = label[0, :, :]
                elif train_or_test == 1:  # During prediction, CSG is conditioned using the SR's next timestep speed
                    speed = fake_ped_speed[id + 1, :, :]
                    if MULTI_CONDITIONAL_MODEL:
                        curr_label = label[0, :, :]
                else:  # During Simulation, CSG is conditioned using the user-defined speed.
                    if SINGLE_CONDITIONAL_MODEL:
                        speed = speed_control(pred_ped_speed[0, :, :], seq_start_end, id=id+1)
                    elif MULTI_CONDITIONAL_MODEL:
                        curr_label = label[0, :, :]
                        speed = speed_control(pred_ped_speed[0, :, :], seq_start_end, label=curr_label)
            if MULTI_CONDITIONAL_MODEL:
                decoder_input = torch.cat([rel_pos, speed, curr_label], dim=1)
            else:
                decoder_input = torch.cat([rel_pos, speed], dim=1)
            decoder_input = self.spatial_embedding(decoder_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)

            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel


class PoolingModule(nn.Module):
    """Pooling module component similar to Social-GAN"""

    def __init__(self, h_dim, mlp_input_dim):
        super(PoolingModule, self).__init__()
        self.mlp_dim = MLP_DIM
        self.h_dim = h_dim
        self.bottleneck_dim = BOTTLENECK_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.mlp_input_dim = mlp_input_dim

        mlp_pre_dim = self.embedding_dim + self.h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, BOTTLENECK_DIM]

        self.pos_embedding = nn.Linear(2, EMBEDDING_DIM)
        self.mlp_pre_pool = make_mlp(mlp_pre_pool_dims, activation=ACTIVATION_RELU, batch_norm=BATCH_NORM, dropout=DROPOUT)

    def forward(self, h_states, seq_start_end, train_or_test, last_pos, label=None):
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden_ped = h_states.view(-1, self.h_dim)[start:end]
            repeat_hstate = curr_hidden_ped.repeat(num_ped, 1).view(num_ped, num_ped, -1)

            feature = last_pos[start:end]
            curr_end_pos_1 = feature.repeat(num_ped, 1)
            curr_end_pos_2 = feature.unsqueeze(dim=1).repeat(1, num_ped, 1).view(-1, 2)
            social_features = curr_end_pos_1[:, :2] - curr_end_pos_2[:, :2]
            position_feature_embedding = self.pos_embedding(social_features.contiguous().view(-1, 2))
            pos_mlp_input = torch.cat(
                [repeat_hstate.view(-1, self.h_dim), position_feature_embedding.view(-1, self.embedding_dim)], dim=1)
            pos_attn_h = self.mlp_pre_pool(pos_mlp_input)
            curr_pool_h = pos_attn_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class AggregationModule(nn.Module):
    """Aggregation module to aggregate 'N' nearest neighbours hidden states"""

    def __init__(self, h_dim, mlp_input_dim):
        super(AggregationModule, self).__init__()
        self.mlp_dim = MLP_DIM
        self.h_dim = h_dim
        self.bottleneck_dim = BOTTLENECK_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.mlp_input_dim = mlp_input_dim

        mlp_pre_dim = self.h_dim * MAX_CONSIDERED_PED
        mlp_pre_pool_dims = [mlp_pre_dim, 512, BOTTLENECK_DIM]

        self.pos_embedding = nn.Linear(2, EMBEDDING_DIM)
        self.mlp_pre_pool = make_mlp(mlp_pre_pool_dims, activation=ACTIVATION_RELU, batch_norm=BATCH_NORM, dropout=DROPOUT)

    def forward(self, h_states, seq_start_end, train_or_test, last_pos, label=None):
        agg_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden_ped = h_states.view(-1, self.h_dim)[start:end]

            feature = last_pos[start:end].cpu().data.numpy()
            dist = squareform(pdist(feature, metric="euclidean"))
            idx = np.argsort(dist)
            req_h_states = []
            for ids in idx:
                if USE_GPU:
                    req_ids = torch.from_numpy(ids).type(torch.cuda.FloatTensor).view(num_ped, 1)
                else:
                    req_ids = torch.from_numpy(ids).type(torch.float).view(num_ped, 1)
                new_h_states = torch.cat([curr_hidden_ped, req_ids], dim=1)
                sorted = new_h_states[new_h_states[:, -1].sort()[1]]
                required_h_states = sorted[:, :-1].contiguous().view(1, -1)
                if num_ped >= MAX_CONSIDERED_PED:
                    req_h_states.append(required_h_states[:, :(MAX_CONSIDERED_PED*self.h_dim)])
                else:
                    if USE_GPU:
                        h_state_zeros = torch.zeros(1, self.h_dim * MAX_CONSIDERED_PED).cuda()
                    else:
                        h_state_zeros = torch.zeros(1, self.h_dim * MAX_CONSIDERED_PED)
                    h_state_zeros[:, :self.h_dim * num_ped] = required_h_states
                    req_h_states.append(h_state_zeros)
            aggregated_h_states = torch.cat(req_h_states, dim=0)
            curr_agg_h = self.mlp_pre_pool(aggregated_h_states)
            agg_h.append(curr_agg_h)
        agg_h = torch.cat(agg_h, dim=0)
        return agg_h


class AttentionModule(nn.Module):
    """Attention module to identify the important agents in the 'N' nearest neighbours"""

    def __init__(self, h_dim, mlp_input_dim):
        super(AttentionModule, self).__init__()
        self.mlp_dim = MLP_DIM
        self.h_dim = h_dim
        self.bottleneck_dim = BOTTLENECK_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.mlp_input_dim = mlp_input_dim

        mlp_pre_dim = self.h_dim + self.embedding_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, BOTTLENECK_DIM]
        self.attn = nn.Linear(MAX_CONSIDERED_PED*BOTTLENECK_DIM, MAX_CONSIDERED_PED)

        self.pos_embedding = nn.Linear(2, EMBEDDING_DIM)
        self.mlp_pre_pool = make_mlp(mlp_pre_pool_dims, activation=ACTIVATION_RELU, batch_norm=BATCH_NORM, dropout=DROPOUT)

    def forward(self, h_states, seq_start_end, train_or_test, last_pos, label=None):
        f_attn_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden_ped = h_states.view(-1, self.h_dim)[start:end]
            repeat_hstate = curr_hidden_ped.repeat(num_ped, 1).view(num_ped, num_ped, -1)
            feature = last_pos[start:end]
            curr_end_pos_1 = feature.repeat(num_ped, 1)
            curr_end_pos_2 = feature.unsqueeze(dim=1).repeat(1, num_ped, 1).view(-1, 2)
            social_features = curr_end_pos_1[:, :2] - curr_end_pos_2[:, :2]
            feature = last_pos[start:end].cpu().data.numpy()
            dist = squareform(pdist(feature, metric="euclidean"))
            idx = np.argsort(dist)
            req_h_states = []
            social_features = social_features.view(num_ped, num_ped, 2)
            if num_ped < MAX_CONSIDERED_PED:
                social_feature_embedding = self.pos_embedding(social_features.contiguous().view(-1, 2))
                if USE_GPU:
                    h_state_zeros = torch.zeros(num_ped, MAX_CONSIDERED_PED, self.h_dim).cuda()
                    feature_zeros = torch.zeros(num_ped, MAX_CONSIDERED_PED, self.embedding_dim).cuda()
                else:
                    h_state_zeros = torch.zeros(num_ped, MAX_CONSIDERED_PED, self.h_dim)
                    feature_zeros = torch.zeros(num_ped, MAX_CONSIDERED_PED, self.embedding_dim)
                h_state_zeros[:num_ped, :num_ped, :] = repeat_hstate.view(num_ped, num_ped, self.h_dim)
                feature_zeros[:num_ped, :num_ped, :] = social_feature_embedding.view(num_ped, num_ped, self.embedding_dim)
                concat_features = torch.cat([h_state_zeros.view(-1, self.h_dim), feature_zeros.view(-1, self.embedding_dim)], dim=1)
            else:
                for ids, curr_features, curr_h_states in zip(idx, social_features, repeat_hstate):
                    if USE_GPU:
                        req_ids = torch.from_numpy(ids).type(torch.cuda.FloatTensor).view(num_ped, 1)
                    else:
                        req_ids = torch.from_numpy(ids).type(torch.float).view(num_ped, 1)
                    new_h_states = torch.cat([curr_h_states, req_ids], dim=1)
                    new_features = torch.cat([curr_features, req_ids], dim=1)
                    h_states_sorted = new_h_states[new_h_states[:, -1].sort()[1]][:MAX_CONSIDERED_PED, :]
                    features_sorted = new_features[new_features[:, -1].sort()[1]][:MAX_CONSIDERED_PED, :]
                    required_h_states = h_states_sorted[:, :-1]
                    required_features = features_sorted[:, :-1]
                    social_feature_embedding = self.pos_embedding(required_features.contiguous().view(-1, 2))
                    req_h_states.append(torch.cat([required_h_states, social_feature_embedding], dim=1))
                concat_features = torch.stack(req_h_states, dim=0)
            attn_h = self.mlp_pre_pool(concat_features.view(-1, (self.h_dim+self.embedding_dim)))
            attn_h = attn_h.view(num_ped, MAX_CONSIDERED_PED, -1)
            attn_w = F.softmax(self.attn(attn_h.view(num_ped, -1)), dim=1)
            attn_w = attn_w.view(num_ped, MAX_CONSIDERED_PED, 1)
            attn_h = torch.sum(attn_h * attn_w, dim=1)
            f_attn_h.append(attn_h)
        f_attn_h = torch.cat(f_attn_h, dim=0)
        return f_attn_h


def speed_control(pred_traj_first_speed, seq_start_end, label=None, id=None):
    """Method that acts as Speed controller: user speed between 0 and 1 is scaled respectively according to Single/Multi condition"""
    for _, (start, end) in enumerate(seq_start_end):
        start = start.item()
        end = end.item()
        if MULTI_CONDITIONAL_MODEL:
            av_tensor = [1, 0, 0]
            av = torch.FloatTensor(av_tensor)
            other_tensor = [0, 1, 0]
            other = torch.FloatTensor(other_tensor)
            agent_tensor = [0, 0, 1]
            agent = torch.FloatTensor(agent_tensor)
            if DIFFERENT_SPEED_MULTI_CONDITION:
                # Implementing different speeds to different agent. To provide constant speed, provide same value for all agents
                for a in range(start, end):
                    for b, c in zip(label[start: end], range(start, end)):
                        if torch.all(torch.eq(b, av)):
                            pred_traj_first_speed[c] = sigmoid(AV_SPEED * AV_MAX_SPEED)
                        elif torch.all(torch.eq(b, other)):
                            pred_traj_first_speed[c] = sigmoid(OTHER_SPEED * OTHER_MAX_SPEED)
                        elif torch.all(torch.eq(b, agent)):
                            pred_traj_first_speed[c] = sigmoid(AGENT_SPEED * AGENT_MAX_SPEED)
        elif SINGLE_CONDITIONAL_MODEL:
            if CONSTANT_SPEED_SINGLE_CONDITION:
                # Implementing constant speed to all agents
                speed_to_simulate = SINGLE_AGENT_MAX_SPEED * CS_SINGLE_CONDITION
                for a in range(start, end):
                    pred_traj_first_speed[a] = sigmoid(speed_to_simulate)

            elif STOP_PED_SINGLE_CONDITION:
                # To stop all pedestrians
                for a in range(start, end):
                    pred_traj_first_speed[a] = sigmoid(0)

    return pred_traj_first_speed.view(-1, 1)


class TrajectoryGenerator(nn.Module):
    def __init__(self, mlp_dim, h_dim):
        super(TrajectoryGenerator, self).__init__()

        self.mlp_dim = MLP_DIM
        self.h_dim = h_dim

        self.mlp_input_dim = mlp_dim
        self.h_dim = h_dim

        self.embedding_dim = EMBEDDING_DIM
        self.noise_dim = NOISE_DIM
        self.num_layers = NUM_LAYERS
        self.bottleneck_dim = BOTTLENECK_DIM

        self.encoder = Encoder(h_dim=h_dim, mlp_input_dim=mlp_dim)
        self.decoder = Decoder(h_dim = h_dim, mlp_input_dim=mlp_dim)

        self.noise_first_dim = NOISE_DIM[0]

        if AGGREGATION_TYPE == 'pooling':
            self.conditionalPoolingModule = PoolingModule(h_dim=h_dim, mlp_input_dim=mlp_dim)
            mlp_decoder_context_dims = [h_dim + BOTTLENECK_DIM, MLP_DIM, h_dim - self.noise_first_dim]
        elif AGGREGATION_TYPE == 'concat':
            self.aggregation_module = AggregationModule(h_dim=h_dim, mlp_input_dim=mlp_dim)
            mlp_decoder_context_dims = [h_dim + BOTTLENECK_DIM, MLP_DIM, h_dim - self.noise_first_dim]
        elif AGGREGATION_TYPE == 'attention':
            self.attention_module = AttentionModule(h_dim=h_dim, mlp_input_dim=mlp_dim)
            mlp_decoder_context_dims = [h_dim + BOTTLENECK_DIM, MLP_DIM, h_dim - self.noise_first_dim]
        else:
            mlp_decoder_context_dims = [h_dim, MLP_DIM, h_dim - self.noise_first_dim]

        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims, activation=ACTIVATION_RELU, batch_norm=BATCH_NORM,
                                            dropout=DROPOUT)

    def add_noise(self, _input, seq_start_end):
        noise_shape = (seq_start_end.size(0),) + self.noise_dim
        if USE_GPU:
            z_decoder = torch.randn(*noise_shape).cuda()
        else:
            z_decoder = torch.randn(*noise_shape)
        _list = []
        for idx, (start, end) in enumerate(seq_start_end):
            noise = z_decoder[idx].view(1, -1).repeat(end.item() - start.item(), 1)
            _list.append(torch.cat([_input[start:end], noise], dim=1))
        decoder_h = torch.cat(_list, dim=0)
        return decoder_h

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj, train_or_test, fake_ped_speed, obs_obj_rel_speed, obs_label=None, pred_label=None, user_noise=None):
        batch = obs_traj_rel.size(1)
        # For multi condition, the encoder is additionally conditioned with agent-type information
        if MULTI_CONDITIONAL_MODEL:
            final_encoder_h = self.encoder(obs_traj_rel, obs_ped_speed, label=obs_label)
        else:
            final_encoder_h = self.encoder(obs_traj_rel, obs_ped_speed, label=None)
        # Aggregation module to jointly reason agent-agent interaction.
        if AGGREGATION_TYPE == 'pooling':
            pm_final_vector = self.conditionalPoolingModule(final_encoder_h, seq_start_end, train_or_test, obs_traj[-1, :, :])
            mlp_decoder_context_input = torch.cat([final_encoder_h.view(-1, self.h_dim), pm_final_vector], dim=1)
        elif AGGREGATION_TYPE == 'concat':
            agg_final_vector = self.aggregation_module(final_encoder_h, seq_start_end, train_or_test, obs_traj[-1, :, :])
            mlp_decoder_context_input = torch.cat([final_encoder_h.view(-1, self.h_dim), agg_final_vector], dim=1)
        elif AGGREGATION_TYPE == 'attention':
            attn_final_vector = self.attention_module(final_encoder_h, seq_start_end, train_or_test, obs_traj[-1, :, :])
            mlp_decoder_context_input = torch.cat([final_encoder_h.view(-1, self.h_dim), attn_final_vector], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(-1, self.h_dim)
        noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        # We add Gaussian noise to induce randomness
        decoder_h = self.add_noise(noise_input, seq_start_end).unsqueeze(dim=0)
        if USE_GPU:
            decoder_c = torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        else:
            decoder_c = torch.zeros(self.num_layers, batch, self.h_dim)

        state_tuple = (decoder_h, decoder_c)

        if MULTI_CONDITIONAL_MODEL:
            decoder_out = self.decoder(obs_traj[-1], obs_traj_rel[-1], state_tuple, seq_start_end, pred_ped_speed,
            train_or_test, fake_ped_speed, label=pred_label)
        else:
            decoder_out = self.decoder(obs_traj[-1], obs_traj_rel[-1], state_tuple, seq_start_end, pred_ped_speed,
            train_or_test, fake_ped_speed)
        pred_traj_fake_rel = decoder_out

        return pred_traj_fake_rel, decoder_h.view(-1, self.h_dim)


class TrajectoryDiscriminator(nn.Module):
    def __init__(self, h_dim, mlp_dim):
        super(TrajectoryDiscriminator, self).__init__()

        self.encoder = Encoder(h_dim, mlp_input_dim=mlp_dim)

        real_classifier_dims = [h_dim, MLP_DIM, 1]
        self.real_classifier = make_mlp(real_classifier_dims, activation=ACTIVATION_RELU, batch_norm=BATCH_NORM, dropout=DROPOUT)

    def forward(self, traj, traj_rel, ped_speed, label=None):
        # Similar to G, the encoder in D for multi-agent model is additionally conditioned on agent-type
        if MULTI_CONDITIONAL_MODEL:
            final_h = self.encoder(traj_rel, ped_speed, label=label)
        else:
            final_h = self.encoder(traj_rel, ped_speed, label=None)
        scores = self.real_classifier(final_h.squeeze())
        return scores
