import torch

from VerifyOutputSpeed import verify_speed
from trajectories import data_loader
from models import TrajectoryGenerator, SpeedEncoderDecoder
from utils import displacement_error, final_displacement_error, relative_to_abs
from constants import *
import numpy as np

from scipy.spatial.distance import pdist, squareform


def collisionPercentage(traj, sequences):
    collided_or_not = []
    for (start, end) in sequences:
        curr_Traj = traj[:, start:end, :].cpu().data.numpy()
        curr_collided_peds = 0
        peds = 0
        for trajectories in curr_Traj:
            peds += trajectories.shape[0]
            dist = squareform(pdist(trajectories, metric="euclidean"))
            np.fill_diagonal(dist, np.nan)
            for rows in dist:
                if any(i <= 0.1 for i in rows):
                    curr_collided_peds += 1

        percentage_of_collision_in_curr_frame = curr_collided_peds / peds
        collided_or_not.append(percentage_of_collision_in_curr_frame)

    collision = sum(collided_or_not) / len(collided_or_not)

    return torch.tensor(collision)


def evaluate_helper(error, traj, seq_start_end):
    sum_ = []
    curr_best_traj = []
    for (start, end) in seq_start_end:
        sum_.append(torch.min(torch.sum(error[start.item():end.item()], dim=0)))
        idx = torch.argmin(torch.sum(error[start.item():end.item()], dim=0))
        curr_best_traj.append(traj[idx, :, start:end, :])
    return torch.cat(curr_best_traj, dim=1), sum(sum_)


def evaluate(loader, generator, num_samples, speed_regressor):
    ade_outer, fde_outer, simulated_output, total_traj, sequences, labels, observed_traj = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if USE_GPU:
                batch = [tensor.cuda() for tensor in batch]
            else:
                batch = [tensor for tensor in batch]
            if MULTI_CONDITIONAL_MODEL:
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed,
                 pred_ped_speed, obs_label, pred_label, obs_obj_rel_speed) = batch
            else:
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed,
                 pred_ped_speed, obs_obj_rel_speed) = batch

            ade, fde, traj_op, traj_obs = [], [], [], []
            total_traj.append(pred_traj_gt.size(1))
            sequences.append(seq_start_end)
            if MULTI_CONDITIONAL_MODEL:
                labels.append(torch.cat([obs_label, pred_label], dim=0))

            for _ in range(num_samples):
                if TEST_METRIC == 1:  # USED DURING PREDICTION ENVIRONMENT
                    if MULTI_CONDITIONAL_MODEL:
                        _, final_enc_h = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                                   pred_traj_gt, 0, None, obs_obj_rel_speed, obs_label=obs_label,
                                                   pred_label=pred_label)
                        fake_speed = speed_regressor(obs_ped_speed, final_enc_h)
                        pred_traj_fake_rel, _ = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed,
                                                          pred_ped_speed,
                                                          pred_traj_gt,
                                                          TEST_METRIC, fake_speed, obs_obj_rel_speed,
                                                          obs_label=obs_label, pred_label=pred_label)
                    else:
                        _, final_enc_h = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                                   pred_traj_gt, 0, None, obs_obj_rel_speed, obs_label=None,
                                                   pred_label=None)
                        fake_speed = speed_regressor(obs_ped_speed, final_enc_h)
                        pred_traj_fake_rel, _ = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed,
                                                          pred_ped_speed,
                                                          pred_traj_gt,
                                                          TEST_METRIC, fake_speed, obs_obj_rel_speed, obs_label=None,
                                                          pred_label=None)
                elif TEST_METRIC == 2:  # Used during Simulation environment
                    if MULTI_CONDITIONAL_MODEL:
                        pred_traj_fake_rel, _ = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed,
                                                          pred_ped_speed,
                                                          pred_traj_gt, TEST_METRIC, None, obs_obj_rel_speed,
                                                          obs_label=obs_label, pred_label=pred_label)
                    else:
                        pred_traj_fake_rel, _ = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed,
                                                          pred_ped_speed,
                                                          pred_traj_gt, TEST_METRIC, None, obs_obj_rel_speed,
                                                          obs_label=None, pred_label=None)

                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))
                fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))
                traj_op.append(pred_traj_fake.unsqueeze(dim=0))
                traj_obs.append(obs_traj.unsqueeze(dim=0))

            best_traj, min_ade_error = evaluate_helper(torch.stack(ade, dim=1), torch.cat(traj_op, dim=0),
                                                       seq_start_end)
            staked_obs = torch.cat(traj_obs, dim=0)
            obs = staked_obs[0]
            observed_traj.append(obs)
            _, min_fde_error = evaluate_helper(torch.stack(fde, dim=1), torch.cat(traj_op, dim=0), seq_start_end)
            ade_outer.append(min_ade_error)
            fde_outer.append(min_fde_error)
            simulated_output.append(best_traj)

        ade = sum(ade_outer) / (sum(total_traj) * PRED_LEN)
        fde = sum(fde_outer) / (sum(total_traj))
        simulated_traj = torch.cat(simulated_output, dim=1)
        total_obs = torch.cat(observed_traj, dim=1).permute(1, 0, 2)
        if MULTI_CONDITIONAL_MODEL:
            all_labels = torch.cat(labels, dim=1)
        last_items_in_sequences = []
        curr_sequences = []
        i = 0
        for sequence_list in sequences:
            last_sequence = sequence_list[-1]
            if i > 0:
                last_items_sum = sum(last_items_in_sequences)
                curr_sequences.append(last_items_sum + sequence_list)
            last_items_in_sequences.append(last_sequence[1])
            if i == 0:
                curr_sequences.append(sequence_list)
                i += 1
                continue

        sequences = torch.cat(curr_sequences, dim=0)
        colpercent = collisionPercentage(simulated_traj, sequences)
        print('Collision Percentage: ', colpercent * 100)

        # The user defined speed is verified by computing inverse sigmoid function on the output speed of the model.
        if TEST_METRIC == 2:
            if SINGLE_CONDITIONAL_MODEL:
                verify_speed(simulated_traj, sequences, labels=None)
            else:
                verify_speed(simulated_traj, sequences, labels=all_labels)

        return ade, fde, colpercent * 100


def main():
    checkpoint = torch.load(CHECKPOINT_NAME)
    if MULTI_CONDITIONAL_MODEL:
        generator = TrajectoryGenerator(mlp_dim=MLP_INPUT_DIM_MULTI_CONDITION,
                                        h_dim=H_DIM_GENERATOR_MULTI_CONDITION)
        speed_regressor = SpeedEncoderDecoder(h_dim=H_DIM_GENERATOR_MULTI_CONDITION)
    else:
        generator = TrajectoryGenerator(mlp_dim=MLP_INPUT_DIM_SINGLE_CONDITION,
                                        h_dim=H_DIM_GENERATOR_SINGLE_CONDITION)
        speed_regressor = SpeedEncoderDecoder(h_dim=H_DIM_GENERATOR_SINGLE_CONDITION)
    generator.load_state_dict(checkpoint['g_state'])
    speed_regressor.load_state_dict(checkpoint['regressor_state'])
    if USE_GPU:
        generator.cuda()
        speed_regressor.cuda()
    generator.train()
    speed_regressor.train()

    if MULTI_CONDITIONAL_MODEL:
        test_dataset = MULTI_TEST_DATASET_PATH
    else:
        test_dataset = SINGLE_TEST_DATASET_PATH
    print('Initializing Test dataset')
    _, loader = data_loader(test_dataset, TEST_METRIC, 'test')
    print('Test dataset preprocessing done')

    cm, ade_final, fde_final = [], [], []
    # For prediction environment, we report the metric for 20 runs
    if TEST_METRIC == 1:
        for _ in range(20):
            ade, fde, ca = evaluate(loader, generator, NUM_SAMPLES, speed_regressor)
            cm.append(ca)
            ade_final.append(ade)
            fde_final.append(fde)
            print(ade, fde)
            print('Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(PRED_LEN, ade, fde))
        print('average collision: ', sum(cm) / len(cm))
        print('average ade: ', sum(ade_final) / len(ade_final))
        print('average fde: ', sum(fde_final) / len(fde_final))
    else:
        ade, fde, ca = evaluate(loader, generator, NUM_SAMPLES, speed_regressor)
        print('Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(PRED_LEN, ade, fde))


if __name__ == '__main__':
    main()
