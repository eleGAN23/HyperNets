import sys, os
import time
import json
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as utils
# from models.SELDNet import Seldnet_vanilla, Seldnet_augmented
from SEDNet import Sednet_vanilla, Sednet_augmented, QSednet_augmented, PHMSednet_augmented, Full_PHMSednet_augmented
from utility_functions import load_model, save_model

import matplotlib.pyplot as plt

'''
Train our baseline model for the Task2 of the L3DAS21 challenge.
This script saves the best model checkpoint, as well as a dict containing
the results (loss and history). To evaluate the performance of the trained model
according to the challenge metrics, please use evaluate_baseline_task2.py.
Command line arguments define the model parameters, the dataset to use and
where to save the obtained results.
'''

def evaluate(model, device, criterion_sed, criterion_doa, dataloader):
    #compute loss without backprop
    model.eval()
    test_loss = 0.
    with tqdm(total=len(dataloader) // args.batch_size) as pbar, torch.no_grad():
        for example_num, (x, target) in enumerate(dataloader):
            target = target.to(device)
            x = x.to(device)
            t = time.time()
            # Compute loss for each instrument/model
            sed = model(x)
            loss = seld_loss(x, target, model, criterion_sed, criterion_doa)
            test_loss += (1. / float(example_num + 1)) * (loss - test_loss)
            pbar.set_description("Current loss: {:.4f}".format(test_loss))
            pbar.update(1)
    return test_loss



def seld_loss(x, target, model, criterion_sed, criterion_doa):
    '''
    compute seld loss as weighted sum of sed (BCE) and doa (MSE) losses
    '''
    target_sed = target[:,:,:args.output_classes*args.class_overlaps]

    #compute loss
    sed = model(x)
    sed = torch.flatten(sed, start_dim=1)
    target_sed = torch.flatten(target_sed, start_dim=1)

    loss_sed = criterion_sed(sed, target_sed) * args.sed_loss_weight

    return loss_sed


def main(args):
    if args.use_cuda:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if args.fixed_seed:
        seed = 1
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    #LOAD DATASET
    print ('\nLoading dataset')

    with open(args.training_predictors_path, 'rb') as f:
        training_predictors = pickle.load(f)
    with open(args.training_target_path, 'rb') as f:
        training_target = pickle.load(f)
    with open(args.validation_predictors_path, 'rb') as f:
        validation_predictors = pickle.load(f)
    with open(args.validation_target_path, 'rb') as f:
        validation_target = pickle.load(f)
    with open(args.test_predictors_path, 'rb') as f:
        test_predictors = pickle.load(f)
    with open(args.test_target_path, 'rb') as f:
        test_target = pickle.load(f)

    training_predictors = np.array(training_predictors)
    training_target = np.array(training_target)
    validation_predictors = np.array(validation_predictors)
    validation_target = np.array(validation_target)
    test_predictors = np.array(test_predictors)
    test_target = np.array(test_target)

    print ('\nShapes:')
    print ('Training predictors: ', training_predictors.shape)
    print ('Validation predictors: ', validation_predictors.shape)
    print ('Test predictors: ', test_predictors.shape)
    print ('Training target: ', training_target.shape)
    print ('Validation target: ', validation_target.shape)
    print ('Test target: ', test_target.shape)
    
    
    # Normalize training predictors with mean 0 and std 1
    train_mag_min = np.mean(training_predictors[:,:8,:,:])
    train_mag_std = np.std(training_predictors[:,:8,:,:])    
    train_phase_min = np.mean(training_predictors[:,8:,:,:])
    train_phase_std = np.std(training_predictors[:,8:,:,:])

    # Input normalization
    training_predictors[:,:8,:,:] -= train_mag_min
    training_predictors[:,:8,:,:] /= train_mag_std
    training_predictors[:,8:,:,:] -= train_phase_min
    training_predictors[:,8:,:,:] /= train_phase_std
    
    # Normalize validation predictors with mean 0 and std 1
    val_mag_min = np.mean(validation_predictors[:,:8,:,:])
    val_mag_std = np.std(validation_predictors[:,:8,:,:])    
    val_phase_min = np.mean(validation_predictors[:,8:,:,:])
    val_phase_std = np.std(validation_predictors[:,8:,:,:])
    validation_predictors[:,:8,:,:] -= val_mag_min
    validation_predictors[:,:8,:,:] /= val_mag_std
    validation_predictors[:,8:,:,:] -= val_phase_min
    validation_predictors[:,8:,:,:] /= val_phase_std

    # Normalize test predictors with mean 0 and std 1
    test_mag_min = np.mean(test_predictors[:,:8,:,:])
    test_mag_std = np.std(test_predictors[:,:8,:,:])    
    test_phase_min = np.mean(test_predictors[:,8:,:,:])
    test_phase_std = np.std(test_predictors[:,8:,:,:])
    test_predictors[:,:8,:,:] -= test_mag_min
    test_predictors[:,:8,:,:] /= test_mag_std
    test_predictors[:,8:,:,:] -= test_phase_min
    test_predictors[:,8:,:,:] /= test_phase_std


    features_dim = int(test_target.shape[-2] * test_target.shape[-1])

    training_predictors = torch.tensor(training_predictors).float()
    validation_predictors = torch.tensor(validation_predictors).float()
    test_predictors = torch.tensor(test_predictors).float()
    training_target = torch.tensor(training_target).float()
    validation_target = torch.tensor(validation_target).float()
    test_target = torch.tensor(test_target).float()
    #build dataset from tensors
    tr_dataset = utils.TensorDataset(training_predictors, training_target)
    val_dataset = utils.TensorDataset(validation_predictors, validation_target)
    test_dataset = utils.TensorDataset(test_predictors, test_target)



    #build data loader from dataset
    tr_data = utils.DataLoader(tr_dataset, args.batch_size, shuffle=True, pin_memory=True)
    val_data = utils.DataLoader(val_dataset, args.batch_size, shuffle=False, pin_memory=True)
    test_data = utils.DataLoader(test_dataset, args.batch_size, shuffle=False, pin_memory=True)

    #LOAD MODEL
    if args.architecture == 'sednet_vanilla':
        n_time_frames = test_predictors.shape[-1]
        model = Sednet_vanilla(time_dim=n_time_frames, freq_dim=args.freq_dim, input_channels=args.input_channels,
                    output_classes=args.output_classes, pool_size=args.pool_size,
                    pool_time=args.pool_time, rnn_size=args.rnn_size, n_rnn=args.n_rnn,
                    fc_size=args.fc_size, dropout_perc=args.dropout_perc,
                    n_cnn_filters=args.n_cnn_filters, class_overlaps=args.class_overlaps,
                    verbose=args.verbose)
    if args.architecture == 'sednet_augmented':
        n_time_frames = test_predictors.shape[-1]
        model = Sednet_augmented(time_dim=n_time_frames, freq_dim=args.freq_dim, input_channels=args.input_channels,
                    output_classes=args.output_classes, pool_size=args.pool_size,
                    pool_time=args.pool_time, rnn_size=args.rnn_size, n_rnn=args.n_rnn,
                    fc_size=args.fc_size, dropout_perc=args.dropout_perc,
                    cnn_filters=args.cnn_filters, class_overlaps=args.class_overlaps,
                    verbose=args.verbose)
    if args.architecture == 'qsednet':
        n_time_frames = test_predictors.shape[-1]
        model = QSednet_augmented(time_dim=n_time_frames, freq_dim=args.freq_dim, input_channels=args.input_channels,
                    output_classes=args.output_classes, pool_size=args.pool_size,
                    pool_time=args.pool_time, rnn_size=args.rnn_size, n_rnn=args.n_rnn,
                    fc_size=args.fc_size, dropout_perc=args.dropout_perc,
                    cnn_filters=args.cnn_filters, class_overlaps=args.class_overlaps,
                    verbose=args.verbose)
    if args.architecture == 'phmsednet_n2':
        n_time_frames = test_predictors.shape[-1]
        model = PHMSednet_augmented(time_dim=n_time_frames, freq_dim=args.freq_dim, input_channels=args.input_channels,
                    output_classes=args.output_classes, pool_size=args.pool_size,
                    pool_time=args.pool_time, rnn_size=args.rnn_size, n_rnn=args.n_rnn,
                    fc_size=args.fc_size, dropout_perc=args.dropout_perc,
                    cnn_filters=args.cnn_filters, class_overlaps=args.class_overlaps,
                    verbose=args.verbose, n=2)
    if args.architecture == 'phmsednet_n4':
        n_time_frames = test_predictors.shape[-1]
        model = PHMSednet_augmented(time_dim=n_time_frames, freq_dim=args.freq_dim, input_channels=args.input_channels,
                    output_classes=args.output_classes, pool_size=args.pool_size,
                    pool_time=args.pool_time, rnn_size=args.rnn_size, n_rnn=args.n_rnn,
                    fc_size=args.fc_size, dropout_perc=args.dropout_perc,
                    cnn_filters=args.cnn_filters, class_overlaps=args.class_overlaps,
                    verbose=args.verbose, n=4)
    if args.architecture == 'phmsednet_n8':
        n_time_frames = test_predictors.shape[-1]
        model = PHMSednet_augmented(time_dim=n_time_frames, freq_dim=args.freq_dim, input_channels=args.input_channels,
                    output_classes=args.output_classes, pool_size=args.pool_size,
                    pool_time=args.pool_time, rnn_size=args.rnn_size, n_rnn=args.n_rnn,
                    fc_size=args.fc_size, dropout_perc=args.dropout_perc,
                    cnn_filters=args.cnn_filters, class_overlaps=args.class_overlaps,
                    verbose=args.verbose, n=8)
    if args.architecture == 'phmsednet_n16':
        n_time_frames = test_predictors.shape[-1]
        model = PHMSednet_augmented(time_dim=n_time_frames, freq_dim=args.freq_dim, input_channels=args.input_channels,
                    output_classes=args.output_classes, pool_size=args.pool_size,
                    pool_time=args.pool_time, rnn_size=args.rnn_size, n_rnn=args.n_rnn,
                    fc_size=args.fc_size, dropout_perc=args.dropout_perc,
                    cnn_filters=args.cnn_filters, class_overlaps=args.class_overlaps,
                    verbose=args.verbose, n=16)
    if args.architecture == 'full_phmsednet_n4':
        n_time_frames = test_predictors.shape[-1]
        model = Full_PHMSednet_augmented(time_dim=n_time_frames, freq_dim=args.freq_dim, input_channels=args.input_channels,
                    output_classes=args.output_classes, pool_size=args.pool_size,
                    pool_time=args.pool_time, rnn_size=args.rnn_size, n_rnn=args.n_rnn,
                    fc_size=args.fc_size, dropout_perc=args.dropout_perc,
                    cnn_filters=args.cnn_filters, class_overlaps=args.class_overlaps,
                    verbose=args.verbose, n=4)

        
        
        
    if args.use_cuda:
        print("Moving model to gpu")
    model = model.to(device)

    #compute number of parameters
    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('Total paramters: ' + str(model_params))

    #set up the loss functions
    criterion_sed = nn.BCELoss()
    criterion_doa = nn.MSELoss()

    #set up optimizer
    optimizer = Adam(params=model.parameters(), lr=args.lr)

    #set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_loss" : np.Inf}

    #load model checkpoint if desired
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = load_model(model, optimizer, args.load_model, args.use_cuda)

    #TRAIN MODEL
    print('TRAINING START')
    train_loss_hist = []
    val_loss_hist = []
    epoch = 1
#     while state["worse_epochs"] < args.patience:
    for curr_epoch in range(args.epochs):
#         print("Training epoch " + str(epoch))
        print("Training epoch " + str(curr_epoch))
        avg_time = 0.
        model.train()
        train_loss = 0.
        with tqdm(total=len(tr_dataset) // args.batch_size) as pbar:
            for example_num, (x, target) in enumerate(tr_data):
                
               
                target = target.to(device)
                x = x.to(device)
                t = time.time()
                # Compute loss for each instrument/model
                optimizer.zero_grad()
                # sed, doa = model(x)
                sed = model(x)
                


                # loss = seld_loss(x, target, model, criterion_sed, criterion_doa)

                target_sed = target[:,:,:args.output_classes*args.class_overlaps]
                sed_flat = torch.flatten(sed, start_dim=1)
                target_sed_flat = torch.flatten(target_sed, start_dim=1)
                loss = criterion_sed(sed_flat, target_sed_flat)

                loss.backward()

                train_loss += (1. / float(example_num + 1)) * (loss - train_loss)
                optimizer.step()
                state["step"] += 1
                t = time.time() - t
                avg_time += (1. / float(example_num + 1)) * (t - avg_time)

                pbar.update(1)

            #PASS VALIDATION DATA
            val_loss = evaluate(model, device, criterion_sed, criterion_doa, val_data)
            print("VALIDATION FINISHED: LOSS: " + str(val_loss))

            # EARLY STOPPING CHECK
            checkpoint_path = args.checkpoint_dir + "/checkpoint_" + args.architecture

            if val_loss >= state["best_loss"]:
                state["worse_epochs"] += 1
            else:
                print("MODEL IMPROVED ON VALIDATION SET!")
                state["worse_epochs"] = 0
                state["best_loss"] = val_loss
                state["best_checkpoint"] = checkpoint_path

                # CHECKPOINT
                print("Saving model...")
                save_model(model, optimizer, state, checkpoint_path)

            state["epochs"] += 1
            #state["worse_epochs"] = 200
            train_loss_hist.append(train_loss.cpu().detach().numpy())
            val_loss_hist.append(val_loss.cpu().detach().numpy())
            epoch += 1

    #LOAD BEST MODEL AND COMPUTE LOSS FOR ALL SETS
    print("TESTING")
    # Load best model based on validation loss
    state = load_model(model, None, state["best_checkpoint"], args.use_cuda)
    #compute loss on all set_output_size
    train_loss = evaluate(model, device, criterion_sed, criterion_doa, tr_data)
    val_loss = evaluate(model, device, criterion_sed, criterion_doa, val_data)
    test_loss = evaluate(model, device, criterion_sed, criterion_doa, test_data)

    #PRINT AND SAVE RESULTS
    results = {'train_loss': train_loss.cpu().detach().numpy(),
               'val_loss': val_loss.cpu().detach().numpy(),
               'test_loss': test_loss.cpu().detach().numpy(),
               'train_loss_hist': train_loss_hist,
               'val_loss_hist': val_loss_hist}

    print ('RESULTS')
    for i in results:
        if 'hist' not in i:
            print (i, results[i])
    out_path = os.path.join(args.results_path, 'results_dict.json')
    np.save(out_path, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #saving/loading parameters
    parser.add_argument('--results_path', type=str, default='RESULTS/Task2',
                        help='Folder to write results dicts into')
    parser.add_argument('--checkpoint_dir', type=str, default='RESULTS/Task2',
                        help='Folder to write checkpoints into')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    #dataset parameters
    parser.add_argument('--training_predictors_path', type=str, default='DATASETS/processed/task2_predictors_train.pkl')
    parser.add_argument('--training_target_path', type=str, default='DATASETS/processed/task2_target_train.pkl')
    parser.add_argument('--validation_predictors_path', type=str, default='DATASETS/processed/task2_predictors_validation.pkl')
    parser.add_argument('--validation_target_path', type=str, default='DATASETS/processed/task2_target_validation.pkl')
    parser.add_argument('--test_predictors_path', type=str, default='DATASETS/processed/task2_predictors_test.pkl')
    parser.add_argument('--test_target_path', type=str, default='DATASETS/processed/task2_target_test.pkl')
    #training parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--use_cuda', type=str, default='True')
    parser.add_argument('--early_stopping', type=str, default='True')
    parser.add_argument('--fixed_seed', type=str, default='False')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--batch_size', type=int, default=3,
                        help="Batch size")
    parser.add_argument('--sr', type=int, default=32000,
                        help="Sampling rate")
    parser.add_argument('--patience', type=int, default=100,
                        help="Patience for early stopping on validation set")
    #model parameters
    #the following parameters produce a prediction for each 100-msecs frame
    parser.add_argument('--architecture', type=str, default='seldnet_augmented',
                        help="model's architecture, [seldnet_vanilla, seldnet_augmented, qseldnet, phmseldnet_n2, phmseldnet_n4, phmseldnet_n8]")
    parser.add_argument('--input_channels', type=int, default=8,
                        help="4/8 for 1/2 mics, multiply x2 if using also phase information")
    parser.add_argument('--class_overlaps', type=int, default=3,
                        help= 'max number of simultaneous sounds of the same class')

    parser.add_argument('--time_dim', type=int, default=4800)
    parser.add_argument('--freq_dim', type=int, default=256)
    parser.add_argument('--output_classes', type=int, default=14)
    parser.add_argument('--pool_size', type=str, default='[[8,2],[8,2],[2,2],[1,1]]')
    parser.add_argument('--cnn_filters', type=str, default='[64,128,256,512]',
                        help= 'only for seldnet augmented')
    parser.add_argument('--pool_time', type=str, default='True')
    parser.add_argument('--rnn_size', type=int, default=256)
    parser.add_argument('--n_rnn', type=int, default=3)
    parser.add_argument('--fc_size', type=int, default=1024)
    parser.add_argument('--dropout_perc', type=float, default=0.3)
    parser.add_argument('--n_cnn_filters', type=float, default=64,
                        help= 'only for seldnet vanilla')
    parser.add_argument('--verbose', type=str, default='False')
    parser.add_argument('--sed_loss_weight', type=float, default=1.)
    parser.add_argument('--doa_loss_weight', type=float, default=5.)

    args = parser.parse_args()

    #eval string bools and lists
    args.use_cuda = eval(args.use_cuda)
    args.early_stopping = eval(args.early_stopping)
    args.fixed_seed = eval(args.fixed_seed)
    args.pool_size= eval(args.pool_size)
    args.pool_time = eval(args.pool_time)
    args.cnn_filters = eval(args.cnn_filters)
    args.verbose = eval(args.verbose)

    main(args)
