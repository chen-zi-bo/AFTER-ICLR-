import importlib
import torch


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'AFTER': 'models.AFTER',
            'Autoformer': 'models.Autoformer',
            'Crossformer': 'models.Crossformer',
            'DLinear': 'models.DLinear',
            'ETSformer': 'models.ETSformer',
            'FEDformer': 'models.FEDformer',
            'FiLM': 'models.FiLM',
            'FreTS': 'models.FreTS',
            'Informer': 'models.Informer',
            'PatchTST': 'models.PatchTST',
            'iTransformer': 'models.iTransformer',
            'Koopa': 'models.Koopa',
            'LightTS': 'models.LightTS',
            'MICN': 'models.MICN',
            'Nonstationary_Transformer': 'models.Nonstationary_Transformer',
            'Pyraformer': 'models.Pyraformer',
            'Reformer': 'models.Reformer',
            'SegRNN': 'models.SegRNN',
            'TiDE': 'models.TiDE',
            'TimeMixer': 'models.TimeMixer',
            'TimesNet': 'models.TimesNet',
            'Transformer': 'models.Transformer',
            'TSMixer': 'models.TSMixer',
        }
        self._model_module_cache = {}
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{args.gpu}')
        else:
            self.device = torch.device('cpu')

    def get_model_module(self, model_name):
        if model_name not in self.model_dict:
            raise ValueError(f'Unknown model: {model_name}')

        if model_name not in self._model_module_cache:
            self._model_module_cache[model_name] = importlib.import_module(self.model_dict[model_name])
        return self._model_module_cache[model_name]

    def _build_model(self):
        raise NotImplementedError
        return None

    # def _acquire_device(self):
    #     if self.args.use_gpu:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = str(
    #             self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
    #         device = torch.device('cuda:{}'.format(self.args.gpu))
    #         print('Use GPU: cuda:{}'.format(self.args.gpu))
    #     else:
    #         device = torch.device('cpu')
    #         print('Use CPU')
    #     return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
