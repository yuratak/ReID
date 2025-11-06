# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
# from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .veri import VeRi
from .vrai_veri import VRAI_VERI
from .vrai import VRAI
from .vrai_test import VRAI_TEST
from .kpneuma import KPNEUMA
from .dataset_loader import ImageDataset

__factory = {
    #'market1501': Market1501,
    # 'cuhk03': CUHK03,
    #'dukemtmc': DukeMTMCreID,
    #'msmt17': MSMT17,
    #'veri': VeRi,
    #'vrai_veri': VRAI_VERI,
    'vrai': VRAI,
    #'vrai_test': VRAI_TEST,
    'kpneuma': KPNEUMA
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
