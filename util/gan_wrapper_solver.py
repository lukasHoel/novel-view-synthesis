"""
Performs training, validation, testing for nvs_model.py with a GAN and calculates loss and saves it to tensorboard.

Author: Lukas Hoellein
"""

import numpy as np

from util.nvs_solver import NVS_Solver
from models.gan.gan_loss import DiscriminatorLoss

import torch
from torch.utils.tensorboard import SummaryWriter
from time import time
from tqdm.auto import tqdm

class GAN_Wrapper_Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self,
                 optim_d=torch.optim.Adam,
                 optim_d_args={},
                 optim_g=torch.optim.Adam,
                 optim_g_args={},
                 extra_args={},
                 log_dir=None,
                 init_discriminator_weights=True):
        """

        Parameters
        ----------
        optim_d: which optimizer to use for discriminator, e.g. Adam
        optim_d_args: see also default_adam_args: specify here valid dictionary of arguments for chosen optimizer
        optim_g: which optimizer to use for generator, e.g. Adam
        optim_g_args: see also default_adam_args: specify here valid dictionary of arguments for chosen optimizer
        extra_args: extra_args that should be used when logging to tensorboard (e.g. model hyperparameters)
        log_dir: where to log to tensorboard
        init_discriminator_weights: if weights of the discriminator should be initialized
        """
        optim_d_args_merged = self.default_adam_args.copy()
        optim_d_args_merged.update(optim_d_args)
        self.optim_d_args = optim_d_args_merged
        self.optim_d = optim_d

        self.writer = SummaryWriter(log_dir)

        self.netD = DiscriminatorLoss(optim_d_args['lr'], init=init_discriminator_weights) # todo other arguments?

        self.optimizer_D = self.optim_d(
            filter(lambda p: p.requires_grad, self.netD.parameters()),
            **self.optim_d_args
        )

        self.nvs_solver = NVS_Solver(optim=optim_g,
                                     optim_args=optim_g_args,
                                     extra_args={},
                                     tensorboard_writer=self.writer)

        nvs_solver_args = {'generator_'+k: v for k, v in self.nvs_solver.hparam_dict.items()}
        self.hparam_dict = {'discriminator': type(self.netD).__name__,
                            'discriminator_optim': self.optim_d.__name__,
                            'discriminator_learning_rate': self.optim_d_args['lr'],
                            **nvs_solver_args,
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

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0, steps=1):
        """
        Train a given nvs_model with the provided data.

        Inputs:
        - model: nvs_model object initialized from nvs_model.py
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        - steps: how many generator/discriminator steps to take before changing to discriminator/generator
        """

        optimizer_G = self.nvs_solver.optim(
            filter(lambda p: p.requires_grad, model.parameters()),
            **self.nvs_solver.optim_args
        )

        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self.netD.to(device)
        weight = 1.0 / float(steps)

        print('START TRAIN on device: {}'.format(device))

        #start = time()
        for epoch in range(num_epochs):  # for every epoch...
            model.train()  # TRAINING mode (for dropout, batchnorm, etc.)
            train_losses = []
            train_accs = []
            for i, sample in enumerate(tqdm(train_loader)):  # for every minibatch in training set
                # RUN GENERATOR steps times
                all_output_images = []
                for j in range(0, steps):
                    nvs_losses, output, _ = self.nvs_solver.forward_pass(model, sample) # todo accuracy is currently _
                    g_losses = self.netD.run_generator_one_step(output["PredImg"], output["OutputImg"])
                    (
                            g_losses["Total Loss"] / weight
                            + nvs_losses["Total Loss"] / weight
                    ).mean().backward()
                    all_output_images += [output]
                optimizer_G.step() # TODO: Why step after loop and not during every loop run?
                optimizer_G.zero_grad()

                # RUN DISCRIMINATOR steps times
                for step in range(0, steps):
                    d_losses = self.netD.run_discriminator_one_step(
                        all_output_images[step]["PredImg"],
                        all_output_images[step]["OutputImg"],
                    )
                    (d_losses["Total Loss"] / weight).mean().backward()
                self.optimizer_D.step()
                self.optimizer_D.zero_grad()

                # UPDATE LOSS: nvs_losses contains G_LOSS and D_LOSS
                # TODO why not combine Total Loss to have nvs_loss + g_loss for logging (gets done for backward pass anyways?
                g_losses.pop("Total Loss")
                d_losses.pop("Total Loss")
                nvs_losses.update(g_losses)
                nvs_losses.update(d_losses)

                # LOGGING of loss
                train_loss = self.nvs_solver.handle_output(nvs_losses, all_output_images[-1], 'Train/', i)
                train_losses.append(train_loss) # TODO is this correct? see above: why not combine everything into Total Loss?
                train_accs.append(0) # todo acc

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

                    nvs_losses, output, _ = self.nvs_solver.forward_pass(model, sample)  # todo accuracy is currently _
                    val_loss = self.nvs_solver.handle_output(nvs_losses, output, 'Val/', i)
                    val_losses.append(val_loss)
                    val_accs.append(0) # todo acc

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
