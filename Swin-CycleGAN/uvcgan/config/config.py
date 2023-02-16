import json
import logging
import os

from uvcgan.consts    import CONFIG_NAME

from .config_base     import ConfigBase
from .data_config     import DataConfig
from .model_config    import ModelConfig
from .transfer_config import TransferConfig

LOGGER = logging.getLogger('uvcgan.config')

class Config(ConfigBase):
    # pylint: disable=too-many-instance-attributes

    __slots__ = [
        'batch_size',
        'data',
        'epochs',
        'image_shape',
        'discriminator',
        'generator',
        'model',
        'model_args',
        'loss',
        'gradient_penalty',
        'seed',
        'scheduler',
        'steps_per_epoch',
        'transfer',
    ]

    def __init__(
        self,
        batch_size       = 32,
        data             = None,
        data_args        = None,
        epochs           = 100,
        image_shape      = (1, 128, 128),
        discriminator    = None,
        generator        = None,
        model            = 'cyclegan',
        model_args       = None,
        loss             = 'lsgan',
        gradient_penalty = None,
        seed             = 0,
        scheduler        = None,
        steps_per_epoch  = 250,
        transfer         = None,
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        self.batch_size      = batch_size
        
        self.model           = model
        self.model_args      = model_args or {}
        self.seed            = seed
        self.loss            = loss
        self.epochs          = epochs
        self.image_shape     = image_shape
        self.scheduler       = scheduler
        self.steps_per_epoch = steps_per_epoch

        if discriminator is not None:
            discriminator = ModelConfig(**discriminator)

        if generator is not None:
            generator = ModelConfig(**generator)

        if gradient_penalty is True:
            gradient_penalty = {}

        if transfer is not None:
            transfer = TransferConfig(**transfer)

        self.discriminator    = discriminator
        self.generator        = generator
        self.gradient_penalty = gradient_penalty
        self.transfer         = transfer

    @staticmethod
    

    def get_savedir(self, outdir, label = None):
        

        discriminator = None
        if self.discriminator is not None:
            discriminator = self.discriminator.model

        generator = None
        if self.generator is not None:
            generator = self.generator.model

        savedir = 'model_d(%s)_m(%s)_d(%s)_g(%s)_%s' % (
            self.data.dataset, self.model, discriminator, generator, label
        )

        savedir = savedir.replace('/', ':')
        path    = os.path.join(outdir, savedir)

        os.makedirs(path, exist_ok = True)
        return path

    def save(self, path):
        # pylint: disable=unspecified-encoding
        with open(os.path.join(path, CONFIG_NAME), 'wt') as f:
            f.write(self.to_json(sort_keys = True, indent = '    '))

    @staticmethod
    def load(path):
        # pylint: disable=unspecified-encoding
        with open(os.path.join(path, CONFIG_NAME), 'rt') as f:
            return Config(**json.load(f))

