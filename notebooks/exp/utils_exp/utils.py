import pandas as pd
import numpy as np
from typing import *

def get_multilabel_y(df:pd.DataFrame, all_doids:List[str], rel_map:Dict, thr_related:int=0.396)->np.ndarray:
    y_train = np.zeros((len(df), len(all_doids)), dtype=int)
    doids = df["doid_id"].to_list()
    for i in range(y_train.shape[0]):
        doid = doids[i]
        if thr_related == 1.0:
            idxs = [all_doids.index(doid)]
        else:
            related = rel_map[thr_related].get(doid, set())
            # add self
            related.add(doid)
            idxs = [all_doids.index(d) for d in related if d in all_doids]
        
        # set positive labels
        y_train[i, idxs] = 1
    return y_train