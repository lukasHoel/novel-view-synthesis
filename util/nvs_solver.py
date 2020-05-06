"""
Performs training, validation, testing for nvs_model.py and calculates loss and saves it to tensorboard.

Author: Lukas Hoellein
"""

import numpy as np

from models.synthesis.synt_loss_metric import SynthesisLoss, QualityMetrics
from util.camera_transformations import invert_K, invert_RT

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from time import time
from tqdm.auto import tqdm

def to_cuda(data_tuple):
    out = ()
    if torch.cuda.is_available():
        for data in data_tuple:
            out += (data.cuda(),)
    return out

def default_batch_loader(batch):
    input_img = batch['image']
    K = batch['cam']['K']
    K_inv = batch['cam']['Kinv']
    input_RT = batch['cam']['RT1']
    input_RT_inv = batch['cam']['RT1inv']
    output_RT = batch['cam']['RT2']
    output_RT_inv = batch['cam']['RT2inv']
    gt_img = batch['output']['image'] if batch['output'] is not None else None
    depth_img = batch['depth']

    return input_img, K, K_inv, input_RT, input_RT_inv, output_RT, output_RT_inv, gt_img, depth_img

# NOTE: Unused, might be used for debugging
def check_norm(img, verbose=False):
    """Try to determine the range of img and return the range in the form of: (left_end, right_end)"""
    max_val = torch.max(img)
    min_val = torch.min(img)

    if verbose:
        print("max_val:", max_val)
        print("min_val:", min_val)

    # Range: [0,1]
    if (0 <= min_val and min_val <= 1) and (0 <= max_val and max_val <= 1):
        return (0,1)

    # Range: [-1,1]
    elif (-1 <= min_val and min_val <= 1) and (-1 <= max_val and max_val <= 1):
        return (-1,1)

    # Range: [0,255]
    elif (0 <= min_val and min_val <= 255) and (0 <= max_val and max_val <= 255):
        return (0,255)

    # Unknown range
    else:
        print("WARNING: Input image doesn't seem to have values in ranges: [0,1], [-1,1], [0,255]")
        return None

# NOTE: Unused, might be used for debugging
def change_norm(img, in_range=None, out_range=[0,1]):
    """Based on the norm scheme of img and output the same image in the new norm scheme""" 
    if not in_range:
        in_range = check_norm(img)

    img = (img - in_range[0]) / (in_range[1] - in_range[0]) * (out_range[1] - out_range[0]) + out_range[0]
    return img

class NVS_Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self,
                 optim=torch.optim.Adam,
                 optim_args={},
                 loss_func=None,
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
        self.loss_func = loss_func if loss_func is not None else SynthesisLoss()
        self.acc_func = QualityMetrics() # TODO: test it
        self.batch_loader = default_batch_loader

        self.writer = SummaryWriter(log_dir) if tensorboard_writer is None else tensorboard_writer

        for key in extra_args.keys():
            extra_args[key] = str(extra_args[key])
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
        output = model(*batch)

        loss_dir = self.loss_func(output['PredImg'], output['OutputImg'])
        acc_dir = None
        if self.acc_func is not None:
            acc_dir = self.acc_func(output['PredImg'], output['OutputImg'])
            # TODO test it

        return loss_dir, output, acc_dir

    def log_loss_and_acc(self, loss_dir, acc_dir, prefix, idx): # acc_dir argument needed
        # WRITE LOSSES
        for loss in loss_dir.keys():
            self.writer.add_scalar(prefix + 'Batch/Loss/' + loss,
                                   loss_dir[loss].data.cpu().numpy(),
                                   idx)
        self.writer.flush()

        # WRITE ACCS
        for acc in acc_dir.keys():
            self.writer.add_scalar(prefix + 'Batch/Accuracy/' + acc,
                                   acc_dir[acc].data.cpu().numpy(),
                                   idx)
        return loss_dir['Total Loss'].data.cpu().numpy(), acc_dir["psnr"].data.cpu().numpy() # could also use acc_dir["ssim"]

    def visualize_output(self, output, take_slice=None, tag="image", step=0):
        """
        Generic method for visualizing a single image or a whole batch

        Parameters
        ----------
        output: batch of data, containing input, target, prediction and depth image
        take_slice: two element tuple or list can be specified to take a slice of the batch (default: take whole batch)
        tag: used for grouping images on tensorboard. e.g. "train", "val", "test" etc.
        step: used for stamping epoch or iteration
        """
        # TODO: depth_batch is ignored for the moment, however, if needed, it can also be integrated later on
        input_batch, target_batch, pred_batch, depth_batch = output["InputImg"].cpu(),\
                                                             output["OutputImg"].cpu(),\
                                                             output["PredImg"].cpu(),\
                                                             output["PredDepth"].cpu()
        with torch.no_grad():
            # In case of a single image add one dimension to the beginning to create single image batch
            if len(pred_batch.shape) == 3:
                input_batch = input_batch.unsqueeze(0)
                target_batch = target_batch.unsqueeze(0)
                pred_batch = pred_batch.unsqueeze(0)
            
            if len(pred_batch.shape) != 4:
                print("Only 3D or 4D tensors can be visualized")
                return

            # If slice specified, take a portion of the batch
            if take_slice and (type(take_slice) in (list, tuple)) and (len(take_slice) == 2):
                input_batch = input_batch[take_slice[0], take_slice[1]]
                target_batch = target_batch[take_slice[0], take_slice[1]]
                pred_batch = pred_batch[take_slice[0], take_slice[1]]
                
            # Store vstack of images: [input_batch0, target_batch0, pred_batch0 ...].T on img_lst
            img_lst = torch.Tensor()

            # Run a loop to interleave images in input_batch, target_batch, pred_batch batches
            for i in range(pred_batch.shape[0]):
                # Each iteration pick input image and corresponding target & pred images
                # As we index image from batch, we need to extend the dimension of indexed images with .unsqueeze(0) for vstack
                # Order in img_list defines the layout. 
                # Current layout: input - target - pred at each row
                img_lst = torch.cat((img_lst, input_batch[i].unsqueeze(0), target_batch[i].unsqueeze(0), pred_batch[i].unsqueeze(0)), dim=0)
            
            img_grid = make_grid(img_lst, nrow=3) # Per row, pick three images from the stack 
            # TODO: this idea can be extended, we can even parametrize this
            # TODO: if needed, determine range of values and use make_grid flags: normalize, range

            self.writer.add_image(tag, img_grid, global_step=step) # NOTE: add_image method expects image values in range [0,1]
            self.writer.flush()

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
                loss_dir, output, test_acc_dir = self.forward_pass(model, sample)
                loss, test_acc = self.log_loss_and_acc(loss_dir,
                                                       test_acc_dir,
                                                       test_name,
                                                       i)
                test_losses.append(loss)
                test_accs.append(test_acc)

                # Print loss every log_nth iteration
                if log_nth != 0 and i % log_nth == 0:
                    print("[Iteration {cur}/{max}] TEST loss: {loss}".format(cur=i + 1,
                                                                              max=len(test_loader),
                                                                              loss=loss))
                    self.visualize_output(output, tag="test", step=i)

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

    def train(self,
              model,
              train_loader,
              val_loader,
              num_epochs=10,
              log_nth_iter=1,
              log_nth_epoch=1,
              tqdm_mode='total'):
        """
        Train a given model with the provided data.

        Inputs:
        :param model: nvs_model object initialized from nvs_model.py
        :param train_loader: train data in torch.utils.data.DataLoader
        :param val_loader: val data in torch.utils.data.DataLoader
        :param num_epochs: total number of training epochs
        :param log_nth_iter: log training accuracy and loss every nth iteration. Default 1: meaning "Log everytime", 0 means "never log"
        :param log_nth_epoch: log training accuracy and loss every nth epoch. Default 1: meaning "Log everytime", 0 means "never log"
        :param tqdm_mode:
                'total': tqdm log how long all epochs will take,
                'epoch': tqdm for each epoch how long it will take,
                anything else, e.g. None: do not use tqdm
        """
        optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN on device: {}'.format(device))

        #start = time()
        epochs = range(num_epochs)
        if tqdm_mode == 'total':
            epochs = tqdm(range(num_epochs))
        for epoch in epochs:  # for every epoch...
            model.train()  # TRAINING mode (for dropout, batchnorm, etc.)
            train_losses = []
            train_accs = []

            train_minibatches = train_loader
            if tqdm_mode == 'epoch':
                train_minibatches = tqdm(train_minibatches)
            for i, sample in enumerate(train_minibatches):  # for every minibatch in training set
                # FORWARD PASS --> Loss + acc calculation
                #print("Time until next forward pass (loading from dataloader + backward pass) took: {}".format(time() - start))
                train_loss_dir, train_output, train_acc_dir = self.forward_pass(model, sample)
                #start = time()

                # BACKWARD PASS --> Gradient-Descent update
                self.backward_pass(train_loss_dir, optim)

                # LOGGING of loss and accuracy
                train_loss, train_acc = self.log_loss_and_acc(train_loss_dir,
                                                              train_acc_dir,
                                                              'Train/',
                                                              epoch*iter_per_epoch + i)
                train_losses.append(train_loss)
                train_accs.append(train_acc)

                # Print loss every log_nth iteration
                if log_nth_iter != 0 and i % log_nth_iter == 0:
                    print("[Iteration {cur}/{max}] TRAIN loss: {loss}".format(cur=i + 1,
                                                                              max=iter_per_epoch,
                                                                              loss=train_loss))
                    self.visualize_output(train_output, tag="train", step=i)

            # ONE EPOCH PASSED --> calculate + log mean train accuracy/loss for this epoch
            mean_train_loss = np.mean(train_losses)
            mean_train_acc = np.mean(train_accs)

            self.train_loss_history.append(mean_train_loss)
            self.train_acc_history.append(mean_train_acc)

            self.writer.add_scalar('Epoch/Loss/Train', mean_train_loss, epoch)
            self.writer.add_scalar('Epoch/Accuracy/Train', mean_train_acc, epoch)

            if log_nth_epoch != 0 and epoch % log_nth_epoch == 0:
                print("[EPOCH {cur}/{max}] TRAIN mean acc/loss: {acc}/{loss}".format(cur=epoch + 1,
                                                                                     max=num_epochs,
                                                                                     acc=mean_train_acc,
                                                                                     loss=mean_train_loss))

            # ONE EPOCH PASSED --> calculate + log validation accuracy/loss for this epoch
            model.eval()  # EVAL mode (for dropout, batchnorm, etc.)
            with torch.no_grad():
                val_losses = []
                val_accs = []

                val_minibatches = train_loader
                if tqdm_mode == 'epoch':
                    val_minibatches = tqdm(val_minibatches)
                for i, sample in enumerate(val_minibatches):
                    # FORWARD PASS --> Loss + acc calculation
                    val_loss_dir, val_output, val_acc_dir = self.forward_pass(model, sample)
                    val_loss, val_acc = self.log_loss_and_acc(val_loss_dir,
                                                              val_acc_dir,
                                                              'Val/',
                                                              epoch*iter_per_epoch + i)
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)

                    # Print loss every log_nth iteration
                    if log_nth_iter != 0 and i % log_nth_iter == 0:
                        print("[Iteration {cur}/{max}] Val loss: {loss}".format(cur=i + 1,
                                                                                max=len(val_loader),
                                                                                loss=val_loss))
                        self.visualize_output(val_output, tag="val", step=i)

                mean_val_loss = np.mean(val_losses)
                mean_val_acc = np.mean(val_accs)

                self.val_loss_history.append(mean_val_loss)
                self.val_acc_history.append(mean_val_acc)

                self.writer.add_scalar('Epoch/Loss/Val', mean_val_loss, epoch)
                self.writer.add_scalar('Epoch/Accuracy/Val', mean_val_acc, epoch)
                self.writer.flush()

                if log_nth_epoch != 0 and epoch % log_nth_epoch == 0:
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
