from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.registry import HOOKS

@HOOKS.register_module()
class CustomLoadCheckpointHook(Hook):
    def __init__(self, ignore_keys=None):
        self.ignore_keys = ignore_keys if ignore_keys else []

    def after_load_checkpoint(self, runner: Runner, checkpoint: dict) -> None:
        """Фильтрация весов после загрузки чекпойнта."""
        state_dict = checkpoint['state_dict']
        filtered_state_dict = {}
        for key, value in state_dict.items():
            # Игнорируем ключи, связанные с классификацией
            if any(ignore_key in key for ignore_key in self.ignore_keys):
                continue
            filtered_state_dict[key] = value
        checkpoint['state_dict'] = filtered_state_dict

@HOOKS.register_module()
class CustomFreezeHook(Hook):
    def __init__(self, unfreeze_epoch=5):
        self.unfreeze_epoch = unfreeze_epoch

    def before_train_epoch(self, runner):
        if runner.epoch == self.unfreeze_epoch:
            for name, param in runner.model.named_parameters():
                if 'backbone' in name:
                    param.requires_grad = True
            runner.logger.info(f'Backbone was unfrozen at epoch {runner.epoch}')

    def before_train(self, runner):
        for name, param in runner.model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
        runner.logger.info('Backbone frozen before training.')

