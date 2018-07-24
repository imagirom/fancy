from copy import deepcopy
import inferno.utils.torch_utils as thu
import torch


class WeightedLoss(torch.nn.Module):

    def __init__(self, loss_weights, trainer=None, loss_names=None, split_by_stages=False, enable_logging=True):
        super(WeightedLoss, self).__init__()
        self.loss_weights = loss_weights
        self.trainer = trainer
        if loss_names is None:
            loss_names = [str(i) for i in range(len(loss_weights))]
        self.loss_names = loss_names
        self.n_losses = len(loss_weights)
        self.logging_enabled = False
        self.enable_logging = enable_logging
        self.split_by_stages = split_by_stages

    def forward(self, preds, labels):
        losses = self.get_losses(preds, labels)
        loss = 0
        for i, current in enumerate(losses):
            loss = loss + self.loss_weights[i] * current
        self.save_losses(losses)
        return loss

    def save_losses(self, losses):
        if self.trainer is None:
            return
        if not self.logging_enabled:
            if self.enable_logging:
                self.register_logger(self.trainer.logger, losses)
            else:
                return
        if not self.split_by_stages:
            losses = [loss.mean() for loss in losses]
            for i, current in enumerate(losses):
                self.trainer.update_state(self.get_loss_name(i), thu.unwrap(torch.mean(current)))
        else:
            for i in range(len(losses)):
                for j, current in enumerate(losses[i]):
                    self.trainer.update_state(self.get_loss_name(i) + f'_stage{j}', thu.unwrap(torch.mean(current)))

    def register_logger(self, logger, losses):  # logger should be a tensorboard logger
        if not self.split_by_stages:
            for i in range(self.n_losses):
                logger.observe_state(self.get_loss_name(i, training=True), 'training')
                logger.observe_state(self.get_loss_name(i, training=False), 'validation')
        else:
            for j in range(len(losses[0])):
                for i in range(self.n_losses):
                    logger.observe_state(self.get_loss_name(i, training=True) + f'_stage{j}', 'training')
                    logger.observe_state(self.get_loss_name(i, training=False) + f'_stage{j}', 'validation')

        self.logging_enabled = True

    def get_loss_name(self, i, training=None):
        if training is None:
            assert self.trainer is not None
            assert self.trainer.model_is_defined
            training = self.trainer.model.training
        if training:
            return 'training_' + self.loss_names[i]
        else:
            return 'validation_' + self.loss_names[i]

    def __getstate__(self):  # TODO make this nicer
        """Return state values to be pickled."""
        return {}

        #if self.trainer is None:
        #    return self
        #return WeightedLoss(self.loss_list, self.loss_weights, trainer=None, loss_names=self.loss_names)