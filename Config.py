import os
import sys
import json
import numpy as np


class Config(object):
    """
    The class for dealing with some setting entry
    """

    def __init__(self, cfg_path='./config.json'):
        with open(cfg_path, 'r') as f:
            self.config_dic = json.load(f)

    def __getitem__(self, attr):
        return self.config_dic[attr]

    def get_config_dic(self):
        return self.config_dic


if __name__ == '__main__':
    cfg = Config()
    print(cfg['common_setting'])
    print(cfg.get_config_dic())
