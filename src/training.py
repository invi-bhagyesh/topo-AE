"""Training classes."""
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .datasets.splitting import split_dataset
import numpy as np
from torchsummary import summary
from .models.submodules import Discriminator 


class TrainingLoop():
    """Training a model using a dataset."""

    def __init__(self, model, dataset, n_epochs, batch_size, learning_rate,
                 weight_decay=1e-5, device='cuda', callbacks=None):
        """Training of a model using a dataset and the defined callbacks.

        Args:
            model: AutoencoderModel
            dataset: Dataset
            n_epochs: Number of epochs to train
            batch_size: Batch size
            learning_rate: Learning rate
            callbacks: List of callbacks
        """
        self.model = model
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.callbacks = callbacks if callbacks else []

    def _execute_callbacks(self, hook, local_variables):
        stop = False
        for callback in self.callbacks:
            # Convert return value to bool --> if callback doesn't return
            # anything we interpret it as False
            stop |= bool(getattr(callback, hook)(**local_variables))
        return stop

    def on_epoch_begin(self, local_variables):
        """Call callbacks before an epoch begins."""
        return self._execute_callbacks('on_epoch_begin', local_variables)

    def on_epoch_end(self, local_variables):
        """Call callbacks after an epoch is finished."""
        return self._execute_callbacks('on_epoch_end', local_variables)

    def on_batch_begin(self, local_variables):
        """Call callbacks before a batch is being processed."""
        self._execute_callbacks('on_batch_begin', local_variables)

    def on_batch_end(self, local_variables):
        """Call callbacks after a batch has be processed."""
        self._execute_callbacks('on_batch_end', local_variables)

    # pylint: disable=W0641
    def __call__(self):
        """Execute the training loop."""
        model = self.model
        dataset = self.dataset
        n_epochs = self.n_epochs
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        # summary(model, input_size=(3, 32, 32))

        n_instances = len(dataset)
        # TODO: Currently we drop the last batch as it might not evenly divide
        # the dataset. This is necassary because the surrogate approach does
        # not yet support changes in the batch dimension.
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=True, drop_last=True)
        n_batches = len(train_loader)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate,
            weight_decay=self.weight_decay)
        # GAN: local discriminator and optimizer
        discriminator = Discriminator().to(self.device)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
        # import torch.nn as nn
        # adv_loss_fn = nn.BCEWithLogitsLoss()

        epoch = 1
        for epoch in range(1, n_epochs+1):
            if self.on_epoch_begin(remove_self(locals())):
                break

            for batch, (img, label) in enumerate(train_loader):
                if self.device == 'cuda':
                    img = img.cuda(non_blocking=True)

                self.on_batch_begin(remove_self(locals()))

                # Set model into training mode and compute loss
                model.train()
                loss, loss_components, reconstruction = self.model(img)

                # --- Minimal GAN loss integration ---
                # BCE-based GAN loss (commented out):
                # real_labels = torch.ones(img.size(0), 1, device=self.device)
                # fake_labels = torch.zeros(img.size(0), 1, device=self.device)
                #
                # real_output = discriminator(img)
                # fake_output = discriminator(reconstruction.detach())
                #
                # d_loss_real = adv_loss_fn(real_output, real_labels)
                # d_loss_fake = adv_loss_fn(fake_output, fake_labels)
                # d_loss = (d_loss_real + d_loss_fake) / 2
                #
                # d_optimizer.zero_grad()
                # d_loss.backward(retain_graph=True)
                # d_optimizer.step()
                #
                # g_output = discriminator(reconstruction)
                # g_loss = adv_loss_fn(g_output, real_labels)

                # --- WGAN loss ---
                real_output = discriminator(img)
                fake_output = discriminator(reconstruction.detach())
                d_loss = -(torch.mean(real_output) - torch.mean(fake_output))

                d_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)
                d_optimizer.step()
                # --- WGAN weight clipping ---
                clip_value = 0.01
                for p in discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

                g_output = discriminator(reconstruction)
                g_loss = -torch.mean(g_output)

                loss = loss + g_loss
                loss_components['loss.gan'] = g_loss
                # --- End GAN loss integration ---

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Call callbacks
                self.on_batch_end(remove_self(locals()))

            if self.on_epoch_end(remove_self(locals())):
                break
        return epoch


def remove_self(dictionary):
    """Remove entry with name 'self' from dictionary.

    This is useful when passing a dictionary created with locals() as kwargs.

    Args:
        dictionary: Dictionary containing 'self' key

    Returns:
        dictionary without 'self' key

    """
    del dictionary['self']
    return dictionary

