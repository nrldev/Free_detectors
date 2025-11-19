from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class CustomFreezeHook(Hook):
    def __init__(self, unfreeze_epoch=5):
        self.unfreeze_epoch = unfreeze_epoch

    def before_train_epoch(self, runner):
        print(f'[CustomHook] Эпоха {runner.epoch}')
        if runner.epoch == self.unfreeze_epoch:
            print(f'\n[CustomHook] Размораживаем backbone на эпохе {runner.epoch}')
            # Проверяем, есть ли атрибут module (для DataParallel/DistributedDataParallel)
            model = runner.model.module if hasattr(runner.model, 'module') else runner.model
            backbone = model.backbone
            for param in backbone.parameters():
                param.requires_grad = True