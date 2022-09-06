import inferno.utils.torch_utils as thu
import torch


class WeightedLoss(torch.nn.Module):

    def __init__(self, loss_weights=None, trainer=None, loss_names=None,
                 split_by_stages=False, enable_logging=True):
        super(WeightedLoss, self).__init__()
        self.loss_weights = loss_weights
        self.enable_logging = enable_logging
        if isinstance(loss_weights, collections.Sized) and not isinstance(loss_weights, str):
            self.n_losses = len(loss_weights)
            self.enable_logging = False
        if loss_names is None and loss_weights is not None:
            loss_names = [str(i) for i in range(len(loss_weights))]
        self.loss_names = loss_names
        self.logging_enabled = False
        self.trainer = trainer
        assert not split_by_stages
        self.split_by_stages = split_by_stages

    def forward(self, preds, labels):
        losses = self.get_losses(preds, labels)
        loss = 0
        for i, current in enumerate(losses):
            if self.loss_weights is None:
                weight = 1
            else:
                weight = self.loss_weights[i]
            loss = loss + weight * current
        if self.loss_weights == 'average':
            losses /= len(losses)
        self.save_losses(losses)
        return loss.mean()

    def save_losses(self, losses):
        if self.trainer is None:
            return
        if not self.logging_enabled:
            if self.enable_logging:
                self.register_logger(self.trainer.logger)
            else:
                return
        losses = [loss.detach().mean() for loss in losses]
        for i, current in enumerate(losses):
            self.trainer.update_state(self.get_loss_name(
                i), thu.unwrap(current))

    def register_logger(self, logger):
        for i in range(self.n_losses):
            logger.observe_state(self.get_loss_name(
                i, training=True), 'training')
            logger.observe_state(self.get_loss_name(
                i, training=False), 'validation')

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
        # mydict = dict(self.__dict__)
        # mydict['trainer'] = None
        return {}
