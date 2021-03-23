import torch
import numpy as np
from constants import *
from utils import get_dataset_name


def get_traj(trajectories, sequences, labels=None):
    print("Enter the agent ids to visualize:", sequences)
    seq_start = int(input("Enter the agent id: "))
    seq_end = int(input("Enter the next-agent:"))
    positions = trajectories[:, seq_start:seq_end, :]
    if MULTI_CONDITIONAL_MODEL:
        label = labels[:, seq_start:seq_end, :]
        return positions, label
    else:
        return positions


def get_distance(trajectories):
    euclid_distance = []
    for a, b in zip(trajectories[:, :], trajectories[1:, :]):
        dist = torch.pairwise_distance(a, b)
        dist = dist.cpu().detach().numpy()
        euclid_distance.append(dist.reshape(1, -1))
    euclid_distance = torch.from_numpy(np.concatenate(euclid_distance, axis=0)).type(torch.float)
    return euclid_distance


def inverse_sigmoid(speeds, max_speed=None, labels=None):
    simulated_speed = []
    inv = torch.log((speeds / (1 - speeds)))
    if SINGLE_CONDITIONAL_MODEL:
        print("The current speeds are: ", inv / max_speed)
    else:
        av_tensor = [1, 0, 0]
        av = torch.FloatTensor(av_tensor)
        other_tensor = [0, 1, 0]
        other = torch.FloatTensor(other_tensor)
        agent_tensor = [0, 0, 1]
        agent_id = torch.FloatTensor(agent_tensor)
        for speed, agent in zip(inv, labels[:PRED_LEN-1, :, :]):
            for a, b, in zip(speed, agent):
                if torch.all(torch.eq(b, av)):
                    s = a / AV_MAX_SPEED
                    simulated_speed.append(s.view(1, 1))
                elif torch.all(torch.eq(b, other)):
                    s = a / OTHER_MAX_SPEED
                    simulated_speed.append(s.view(1, 1))
                elif torch.all(torch.eq(b, agent_id)):
                    s = a / AGENT_MAX_SPEED
                    simulated_speed.append(s.view(1, 1))
        simulated_speed = torch.cat(simulated_speed, dim=0)
        print('the labels are: ', labels)
        print("The current speeds are: ", simulated_speed.view(PRED_LEN-1, -1))


def get_speed_from_distance(distance):
    # Since we skip the speed calculation (see trajectories.py for more explanation), we directly pass the distance through sigmoid layer
    if MULTI_CONDITIONAL_MODEL:
        sigmoid_speed = torch.sigmoid(distance)
    else:
        speed = distance / FRAMES_PER_SECOND_SINGLE_CONDITION
        sigmoid_speed = torch.sigmoid(speed)
    return sigmoid_speed


def verify_speed(traj, sequences, labels=None):
    if MULTI_CONDITIONAL_MODEL:
        traj, label = get_traj(traj, sequences, labels=labels)
    else:
        dataset_name = get_dataset_name(SINGLE_TEST_DATASET_PATH)
        traj = get_traj(traj, sequences, labels=None)
    dist = get_distance(traj)
    speed = get_speed_from_distance(dist)
    # We calculate inverse sigmoid to verify the speed
    if MULTI_CONDITIONAL_MODEL:
        inverse_sigmoid(speed, labels=label)
    else:
        maxspeed= SINGLE_AGENT_MAX_SPEED
        inverse_sigmoid(speed, max_speed=maxspeed)
