#!bin/usr/python3
#Jack Zhang cx416

import pandas as pd

SPAM_HEADERS = ['index', 'geoDistance', 'senderHour', 'AverageIPNeighborDistance', 'fngr_wss(K)', 
            'fngr_ttl', 'OS', 'pkts_sunk', 'ptks_ssourced', 'rxmt_sourced', 'rxmt_sunk', 'rsts_sourced', 
            'rsts_sunk', 'fins_sourced', 'fins_sunk', 'idle', '3whs' 'jvar', 'rttv', 'bytecount', 
            'throughput', 'labels'
            ]


VOLCANO_HEADERS = ['chip_index', 'image_id'] 
chip_pixel_values = ['chip_pixel_value_' + str(x) for x in range(1, 226)]
VOLCANO_HEADERS += chip_pixel_values + ['labels']
class datasets:
    def load_spam(self, data_loc):
        data = pd.read_csv(data_loc, names=SPAM_HEADERS)
        y = data['labels']
        x = data
        del x['labels']
        return (x, y)

    def load_volcanoes(self, data_loc):
        data = pd.read_csv(data_loc, names=VOLCANO_HEADERS)
        y = data['labels']
        x = data
        del x['labels']
        return (x, y)

