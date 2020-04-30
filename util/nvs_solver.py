"""
Performs training, validation, testing for nvs_model.py and calculates loss and saves it to tensorboard.

Author: Lukas Hoellein
"""

import numpy as np

from models.synthesis.synthesis_loss import SynthesisLoss
from util.camera_transformations import invert_K, invert_RT

import torch
from torch.utils.tensorboard import SummaryWriter
from time import time
from tqdm.auto import tqdm

def to_cuda(data_tuple):
    out = ()
    if torch.cuda.is_available():
        for data in data_tuple:
            out += data.cuda()
    return out

def default_batch_loader(batch):
    input_img = batch['image']
    K = batch['cam']['K']
    K_inv = invert_K(K)
    input_RT = batch['cam']['RT1']
    input_RT_inv = invert_RT(input_RT)
    output_RT = batch['cam']['RT2']
    output_RT_inv = invert_RT(output_RT)
    gt_img = batch['output']['image']
    depth_img = batch['depth']
    return input_img, K, K_inv, input_RT, input_RT_inv, output_RT, output_RT_inv, gt_img, depth_img

# TODO which accuracy fits? are they even already calculated in SynthesisLoss as other "losses"?
def accuracy(self, scores, y):
    with torch.no_grad():
        _, preds = torch.max(scores, 1) # select highest value as the predicted class
        y_mask = y >= 0 # do not allow "-1" segmentation value
        acc = np.mean((preds == y)[y_mask].data.cpu().numpy())  # check if prediction is correct + average of it for all N inputs
        return acc

class NVS_Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self,
                 optim=torch.optim.Adam,
                 optim_args={},
                 extra_args={},
                 tensorboard_writer=None,
                 log_dir=None):
        """

        Parameters
        ----------
        optim: which optimizer to use, e.g. Adam
        optim_args: see also default_adam_args: specify here valid dictionary of arguments for chosen optimizer
        extra_args: extra_args that should be used when logging to tensorboard (e.g. model hyperparameters)
        tensorboard_writer: instance to use for writing to tensorboard. can be None, then a new one will be created.
        log_dir: where to log to tensorboard. Only used when no tensorboard_writer is given.
        """
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = SynthesisLoss() # todo use custom values?
        self.acc_func = None # todo what acc?
        self.batch_loader = default_batch_loader

        self.writer = SummaryWriter(log_dir) if tensorboard_writer is None else tensorboard_writer
        self.hparam_dict = {'loss_function': type(self.loss_func).__name__,
                            'optimizer': self.optim.__name__,
                            'learning_rate': self.optim_args['lr'],
                            'weight_decay': self.optim_args['weight_decay'],
                            **extra_args}

        print("Hyperparameters of this solver: {}".format(self.hparam_dict))

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def forward_pass(self, model, batch):

        batch = to_cuda(self.batch_loader(batch))
        output = model(batch)

        loss_dir = self.loss_func(output['PredImg'], output['OutputImg'])
        if self.acc_func is not None:
            pass
            # TODO impl

        return loss_dir, output, 0 # TODO acc impl, do not just return 0

    def handle_output(self, loss_dir, output, prefix, idx):
        # WRITE LOSSES
        for loss in loss_dir.keys():
            self.writer.add_scalar(prefix + 'Batch/Loss/' + loss,
                                   loss_dir[loss].data.cpu().numpy(),
                                   idx)
        self.writer.flush()

        # WRITE IMAGES with output
        #TODO see tmp_solver

        #TODO handle accuracy

        return loss_dir['Total Loss'].data.cpu().numpy()


    def test(self, model, test_loader, test_prefix='/', log_nth=0):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        if test_prefix == None:
            test_prefix = '/'
        elif not test_prefix.endswith('/'):
            test_prefix += '/'
        test_name = 'Test/' + test_prefix

        with torch.no_grad():
            test_losses = []
            test_accs = []
            for i, sample in enumerate(tqdm(test_loader)):
                loss_dir, output, test_acc = self.forward_pass(model, sample)
                loss = self.handle_output(loss_dir, output, test_name, i)
                test_losses.append(loss)
                test_accs.append(test_acc)

                # Print loss every log_nth iteration
                if (i % log_nth == 0):
                    print("[Iteration {cur}/{max}] TEST loss: {loss}".format(cur=i + 1,
                                                                              max=len(test_loader),
                                                                              loss=loss))

            mean_loss = np.mean(test_losses)
            mean_acc = np.mean(test_accs)

            self.writer.add_scalar(test_name + 'Mean/Loss', mean_loss, 0)
            self.writer.add_scalar(test_name + 'Mean/Accuracy', mean_acc, 0)
            self.writer.flush()

            print("[TEST] mean acc/loss: {acc}/{loss}".format(acc=mean_acc, loss=mean_loss))

    def backward_pass(self, loss_dir, optim):
        loss_dir['Total Loss'].backward()
        optim.step()
        optim.zero_grad()

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN on device: {}'.format(device))

        #start = time()
        for epoch in range(num_epochs):  # for every epoch...
            model.train()  # TRAINING mode (for dropout, batchnorm, etc.)
            train_losses = []
            train_accs = []
            for i, sample in enumerate(tqdm(train_loader)):  # for every minibatch in training set
                # FORWARD PASS --> Loss + acc calculation
                #print("Time until next forward pass (loading from dataloader + backward pass) took: {}".format(time() - start))
                train_loss_dir, train_output, train_acc = self.forward_pass(model, sample)
                #start = time()

                # BACKWARD PASS --> Gradient-Descent update
                self.backward_pass(train_loss_dir, optim)

                # LOGGING of loss and accuracy
                train_loss = self.handle_output(train_loss_dir, train_output, 'Train/', i)
                train_losses.append(train_loss)
                train_accs.append(train_acc)

                # Print loss every log_nth iteration
                if (i % log_nth == 0):
                    print("[Iteration {cur}/{max}] TRAIN loss: {loss}".format(cur=i + 1,
                                                                              max=iter_per_epoch,
                                                                              loss=train_loss))

            # ONE EPOCH PASSED --> calculate + log mean train accuracy/loss for this epoch
            mean_train_loss = np.mean(train_losses)
            mean_train_acc = np.mean(train_accs)

            self.train_loss_history.append(mean_train_loss)
            self.train_acc_history.append(mean_train_acc)

            self.writer.add_scalar('Epoch/Loss/Train', mean_train_loss, epoch)
            self.writer.add_scalar('Epoch/Accuracy/Train', mean_train_acc, epoch)

            print("[EPOCH {cur}/{max}] TRAIN mean acc/loss: {acc}/{loss}".format(cur=epoch + 1,
                                                                                 max=num_epochs,
                                                                                 acc=mean_train_acc,
                                                                                 loss=mean_train_loss))

            # ONE EPOCH PASSED --> calculate + log validation accuracy/loss for this epoch
            model.eval()  # EVAL mode (for dropout, batchnorm, etc.)
            with torch.no_grad():
                val_losses = []
                val_accs = []
                for i, sample in enumerate(tqdm(val_loader)):
                    # FORWARD PASS --> Loss + acc calculation
                    val_loss_dir, val_output, val_acc = self.forward_pass(model, sample)
                    val_loss = self.handle_output(val_loss_dir, val_output, 'Val/', i)
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)

                    # Print loss every log_nth iteration
                    if (i % log_nth == 0):
                        print("[Iteration {cur}/{max}] Val loss: {loss}".format(cur=i + 1,
                                                                                max=len(val_loader),
                                                                                loss=val_loss))

                mean_val_loss = np.mean(val_losses)
                mean_val_acc = np.mean(val_accs)

                self.val_loss_history.append(mean_val_loss)
                self.val_acc_history.append(mean_val_acc)

                self.writer.add_scalar('Epoch/Loss/Val', mean_val_loss, epoch)
                self.writer.add_scalar('Epoch/Accuracy/Val', mean_val_acc, epoch)
                self.writer.flush()

                print("[EPOCH {cur}/{max}] VAL mean acc/loss: {acc}/{loss}".format(cur=epoch + 1,
                                                                                   max=num_epochs,
                                                                                   acc=mean_val_acc,
                                                                                   loss=mean_val_loss))

        self.writer.add_hparams(self.hparam_dict, {
            'HParam/Accuracy/Val': self.val_acc_history[-1],
            'HParam/Accuracy/Train': self.train_acc_history[-1],
            'HParam/Loss/Val': self.val_loss_history[-1],
            'HParam/Loss/Train': self.train_loss_history[-1]
        })
        self.writer.flush()
        print('FINISH.')