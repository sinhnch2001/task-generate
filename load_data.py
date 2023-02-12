# LOAD DATA ELI5
import json
from lfqa import *


def preprocess_data_eli5(dir_file="C:\ALL\OJT\TASK\data_eli5"):
    eli5_train = json.load(open(dir_file + '\ELI5_train_10_doc.json'))
    eli5_valid = json.load(open(dir_file + '\ELI5_val_10_doc.json'))
    eli5_train = eli5_train[:3]
    eli5_valid = eli5_valid[:3]

    # PRE PROCESSING DOCS
    eli5_train_docs = []
    eli5_valid_docs = []
    for example in eli5_train:
        support_doc = "<P> " + " <P> ".join([p for p in example["ctxs"]])
        eli5_train_docs += [(example['question_id'], support_doc)]
    for example in eli5_valid:
        support_doc = "<P> " + " <P> ".join([p for p in example["ctxs"]])
        eli5_valid_docs += [(example['question_id'], support_doc)]

    # LOAD DOCS JSON for train and valid
    s2s_train_dset = ELI5DatasetS2S(eli5_train, document_cache=dict([(k, d) for k, d in eli5_train_docs]))
    s2s_valid_dset = ELI5DatasetS2S(eli5_valid, document_cache=dict([(k, d) for k, d in eli5_valid_docs]))

    return s2s_train_dset, s2s_valid_dset
