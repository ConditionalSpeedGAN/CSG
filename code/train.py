import gc
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from constants import *


from trajectories import data_loader
from utils import gan_g_loss, gan_d_loss, l2_loss, displacement_error, final_displacement_error, relative_to_abs, mae_loss

from models import TrajectoryGenerator, TrajectoryDiscriminator, SpeedEncoderDecoder

torch.backends.cudnn.benchmark = True


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def main():
    train_metric = 0
    if MULTI_CONDITIONAL_MODEL and SINGLE_CONDITIONAL_MODEL:
        raise ValueError("Please select either Multi conditional model or single conditional model flag in constants.py")
    print("Process Started")
    if SINGLE_CONDITIONAL_MODEL:
        train_path = SINGLE_TRAIN_DATASET_PATH
        val_path = SINGLE_VAL_DATASET_PATH
    else:
        train_path = MULTI_TRAIN_DATASET_PATH
        val_path = MULTI_VAL_DATASET_PATH
    print("Initializing train dataset")
    train_dset, train_loader = data_loader(train_path, train_metric, 'train')
    print("Initializing val dataset")
    _, val_loader = data_loader(val_path, train_metric, 'val')

    if MULTI_CONDITIONAL_MODEL:
        iterations_per_epoch = len(train_dset) / BATCH_MULTI_CONDITION / D_STEPS
        NUM_ITERATIONS = int(iterations_per_epoch * NUM_EPOCHS_MULTI_CONDITION)
        generator = TrajectoryGenerator(mlp_dim=MLP_INPUT_DIM_MULTI_CONDITION,
                                        h_dim=H_DIM_GENERATOR_MULTI_CONDITION)
        discriminator = TrajectoryDiscriminator(mlp_dim=MLP_INPUT_DIM_MULTI_CONDITION,
                                                h_dim=H_DIM_DISCRIMINATOR_MULTI_CONDITION)
        speed_regressor = SpeedEncoderDecoder(h_dim=H_DIM_GENERATOR_MULTI_CONDITION)
        required_epoch = NUM_EPOCHS_MULTI_CONDITION

    elif SINGLE_CONDITIONAL_MODEL:
        iterations_per_epoch = len(train_dset) / BATCH_SINGLE_CONDITION / D_STEPS

        NUM_ITERATIONS = int(iterations_per_epoch * NUM_EPOCHS_SINGLE_CONDITION)
        generator = TrajectoryGenerator(mlp_dim=MLP_INPUT_DIM_SINGLE_CONDITION,
                                        h_dim=H_DIM_GENERATOR_SINGLE_CONDITION)
        discriminator = TrajectoryDiscriminator(mlp_dim=MLP_INPUT_DIM_SINGLE_CONDITION,
                                                h_dim=H_DIM_DISCRIMINATOR_SINGLE_CONDITION)
        speed_regressor = SpeedEncoderDecoder(h_dim=H_DIM_GENERATOR_SINGLE_CONDITION)
        required_epoch = NUM_EPOCHS_SINGLE_CONDITION

    print(iterations_per_epoch)
    generator.apply(init_weights)
    if USE_GPU:
        generator.type(torch.cuda.FloatTensor).train()
    else:
        generator.type(torch.FloatTensor).train()
    print('Here is the generator:')
    print(generator)

    discriminator.apply(init_weights)
    if USE_GPU:
        discriminator.type(torch.cuda.FloatTensor).train()
    else:
        discriminator.type(torch.FloatTensor).train()
    print('Here is the discriminator:')
    print(discriminator)

    speed_regressor.apply(init_weights)
    if USE_GPU:
        speed_regressor.type(torch.cuda.FloatTensor).train()
    else:
        speed_regressor.type(torch.FloatTensor).train()
    print('Here is the Speed Regressor:')
    print(speed_regressor)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(generator.parameters(), lr=G_LEARNING_RATE)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=D_LEARNING_RATE)
    optimizer_speed_regressor = optim.Adam(speed_regressor.parameters(), lr=D_LEARNING_RATE)

    t, epoch = 0, 0
    checkpoint = {
        'G_losses': defaultdict(list),
        'D_losses': defaultdict(list),
        'g_state': None,
        'g_optim_state': None,
        'd_state': None,
        'd_optim_state': None,
        'g_best_state': None,
        'd_best_state': None,
        'best_regressor_state': None,
        'regressor_state': None
    }
    val_ade_list, val_fde_list, train_ade, train_fde, train_avg_speed_error, val_avg_speed_error, val_msae_list = [], [], [], [], [], [], []
    train_ade_list, train_fde_list = [], []

    while epoch < required_epoch:
        gc.collect()
        d_steps_left, g_steps_left, speed_regression_steps_left = D_STEPS, G_STEPS, SR_STEPS
        epoch += 1
        print('Starting epoch {}'.format(epoch))
        disc_loss, gent_loss, sr_loss = [], [], []
        for batch in train_loader:
            if d_steps_left > 0:
                losses_d = discriminator_step(batch, generator, discriminator, d_loss_fn, optimizer_d)
                disc_loss.append(losses_d['D_total_loss'])
                d_steps_left -= 1
            elif g_steps_left > 0:
                losses_g = generator_step(batch, generator, discriminator, g_loss_fn, optimizer_g)
                speed_regression_loss = speed_regressor_step(batch, generator, speed_regressor, optimizer_speed_regressor)
                losses_g['Speed_Regression_Loss'] = speed_regression_loss['Speed_Regression_Loss']
                sr_loss.append(speed_regression_loss['Speed_Regression_Loss'])
                gent_loss.append(losses_g['G_discriminator_loss'])
                g_steps_left -= 1

            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if t > 0 and t % CHECKPOINT_EVERY == 0:

                print('t = {} / {}'.format(t + 1, NUM_ITERATIONS))
                for k, v in sorted(losses_d.items()):
                    print('  [D] {}: {:.3f}'.format(k, v))
                for k, v in sorted(losses_g.items()):
                    print('  [G] {}: {:.3f}'.format(k, v))

                print('Checking stats on val ...')
                metrics_val = check_accuracy(val_loader, generator, discriminator, d_loss_fn, speed_regressor)
                print('Checking stats on train ...')
                metrics_train = check_accuracy(train_loader, generator, discriminator, d_loss_fn, speed_regressor)

                for k, v in sorted(metrics_val.items()):
                    print('  [val] {}: {:.3f}'.format(k, v))
                for k, v in sorted(metrics_train.items()):
                    print('  [train] {}: {:.3f}'.format(k, v))

                val_ade_list.append(metrics_val['ade'])
                val_fde_list.append(metrics_val['fde'])

                train_ade_list.append(metrics_train['ade'])
                train_fde_list.append(metrics_train['fde'])

                if metrics_val.get('ade') == min(val_ade_list) or metrics_val['ade'] < min(val_ade_list) or metrics_val.get('fde') == min(val_fde_list) or metrics_val['fde'] < min(val_fde_list):
                    checkpoint['g_best_state'] = generator.state_dict()
                if metrics_val.get('ade') == min(val_ade_list) or metrics_val['ade'] < min(val_ade_list):
                    print('New low for avg_disp_error')
                    checkpoint['best_g_state'] = generator.state_dict()
                if metrics_val.get('fde') == min(val_fde_list) or metrics_val['fde'] < min(val_fde_list):
                    print('New low for final_disp_error')

                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                checkpoint['regressor_state'] = speed_regressor.state_dict()
                torch.save(checkpoint, CHECKPOINT_NAME)
                print('Done.')

            t += 1
            d_steps_left = D_STEPS
            g_steps_left = G_STEPS
            if t >= NUM_ITERATIONS:
                break


def discriminator_step(batch, generator, discriminator, d_loss_fn, optimizer_d):
    if USE_GPU:
        batch = [tensor.cuda() for tensor in batch]
    else:
        batch = [tensor for tensor in batch]
    if MULTI_CONDITIONAL_MODEL:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed, obs_label, pred_label,
         obs_obj_rel_speed) = batch
        generator_out, _ = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                  pred_traj_gt, TRAIN_METRIC, None, obs_obj_rel_speed, obs_label=obs_label, pred_label=pred_label)
    else:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed, obs_obj_rel_speed) = batch
        generator_out, _ = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                  pred_traj_gt, TRAIN_METRIC, None, obs_obj_rel_speed, obs_label=None, pred_label=None)

    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
    ped_speed = torch.cat([obs_ped_speed, pred_ped_speed], dim=0)
    if MULTI_CONDITIONAL_MODEL:
        label_info = torch.cat([obs_label, pred_label], dim=0)
        scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, label=label_info)
        scores_real = discriminator(traj_real, traj_real_rel, ped_speed, label=label_info)
    else:
        scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, label=None)
        scores_real = discriminator(traj_real, traj_real_rel, ped_speed, label=None)

    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    optimizer_d.step()

    return losses


def speed_regressor_step(batch, generator, speed_regressor, optimizer_speed_regressor):
    losses = {}
    speed_loss = []
    if USE_GPU:
        batch = [tensor.cuda() for tensor in batch]
    else:
        batch = [tensor for tensor in batch]
    if MULTI_CONDITIONAL_MODEL:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed, obs_label, pred_label, obs_obj_rel_speed) = batch
    else:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed, obs_obj_rel_speed) = batch

    if MULTI_CONDITIONAL_MODEL:
        _, final_enc_h = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj_gt, TRAIN_METRIC, None, obs_obj_rel_speed, obs_label=obs_label, pred_label=pred_label)
    else:
        _, final_enc_h = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed, pred_traj_gt, TRAIN_METRIC, None, obs_obj_rel_speed, obs_label=None, pred_label=None)
    fake_ped_speed = speed_regressor(obs_ped_speed, final_enc_h)
    loss_mask = loss_mask[:, OBS_LEN:]
    speed_loss.append(L2_LOSS_WEIGHT * mae_loss(fake_ped_speed, pred_ped_speed, mode='raw', speed_reg='speed_regressor'))
    total_speed_loss = torch.zeros(1).to(pred_ped_speed)
    speed_loss = torch.stack(speed_loss, dim=1)
    for start, end in seq_start_end.data:
        _speed_loss = speed_loss[start:end]
        _speed_loss = torch.sum(_speed_loss, dim=0)
        _speed_loss = torch.min(_speed_loss) / torch.sum(loss_mask[start:end])
        total_speed_loss += _speed_loss
    losses['Speed_Regression_Loss'] = total_speed_loss.item()
    optimizer_speed_regressor.zero_grad()
    total_speed_loss.backward()
    optimizer_speed_regressor.step()
    return losses


def generator_step(batch, generator, discriminator, g_loss_fn, optimizer_g):
    if USE_GPU:
        batch = [tensor.cuda() for tensor in batch]
    else:
        batch = [tensor for tensor in batch]
    if MULTI_CONDITIONAL_MODEL:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed,
        obs_label, pred_label, obs_obj_rel_speed) = batch
    else:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end, obs_ped_speed, pred_ped_speed, obs_obj_rel_speed) = batch

    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, OBS_LEN:]

    for _ in range(BEST_K):
        if MULTI_CONDITIONAL_MODEL:
            generator_out, final_enc_h = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                  pred_traj_gt, TRAIN_METRIC, None, obs_obj_rel_speed, obs_label=obs_label, pred_label=pred_label)
        else:
            generator_out, final_enc_h = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                      pred_traj_gt, TRAIN_METRIC, None, obs_obj_rel_speed, obs_label=None, pred_label=None)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        if L2_LOSS_WEIGHT > 0:
            g_l2_loss_rel.append(L2_LOSS_WEIGHT * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if L2_LOSS_WEIGHT > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
    ped_speed = torch.cat([obs_ped_speed, pred_ped_speed], dim=0)
    if MULTI_CONDITIONAL_MODEL:
        label_info = torch.cat([obs_label, pred_label], dim=0)
        scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, label=label_info)
    else:
        scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, label=None)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    optimizer_g.step()

    return losses


def check_accuracy(loader, generator, discriminator, d_loss_fn, speed_regressor):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, f_disp_error, mean_speed_disp_error, final_speed_disp_error = [], [], [], []
    total_traj = 0
    loss_mask_sum = 0
    generator.eval()
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

            if MULTI_CONDITIONAL_MODEL:
                pred_traj_fake_rel, final_enc_h = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                  pred_traj_gt, TRAIN_METRIC, None, obs_obj_rel_speed, obs_label=obs_label, pred_label=pred_label)
            else:
                pred_traj_fake_rel, final_enc_h = generator(obs_traj, obs_traj_rel, seq_start_end, obs_ped_speed, pred_ped_speed,
                                      pred_traj_gt, TRAIN_METRIC, None, obs_obj_rel_speed, obs_label=None, pred_label=None)

            fake_ped_speed = speed_regressor(obs_ped_speed, final_enc_h)

            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
            loss_mask = loss_mask[:, OBS_LEN:]

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )

            abs_speed_los = cal_mae_speed_loss(pred_ped_speed, fake_ped_speed)
            ade = displacement_error(pred_traj_gt, pred_traj_fake)
            fde = final_displacement_error(pred_traj_gt, pred_traj_fake)

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
            ped_speed = torch.cat([obs_ped_speed, pred_ped_speed], dim=0)
            if MULTI_CONDITIONAL_MODEL:
                label_info = torch.cat([obs_label, pred_label], dim=0)
                scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, label=label_info)
                scores_real = discriminator(traj_real, traj_real_rel, ped_speed, label=label_info)
            else:
                scores_fake = discriminator(traj_fake, traj_fake_rel, ped_speed, label=None)
                scores_real = discriminator(traj_real, traj_real_rel, ped_speed, label=None)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            f_disp_error.append(fde.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            if total_traj >= NUM_SAMPLE_CHECK:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum
    metrics['ade'] = sum(disp_error) / (total_traj * PRED_LEN)
    metrics['fde'] = sum(f_disp_error) / total_traj

    generator.train()
    return metrics


def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel, loss_mask):
    g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode='sum')
    g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum')
    return g_l2_loss_abs, g_l2_loss_rel


def cal_mae_speed_loss(pred_speed_gt, pred_speed_fake):
    g_l2_speed_loss = mae_loss(pred_speed_gt, pred_speed_fake, speed_reg='speed_reg', mode='sum')
    return g_l2_speed_loss


if __name__ == '__main__':
    main()
