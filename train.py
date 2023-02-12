from lfqa import *
from load_data import *


# CREATE ArgumentsS2S
class ArgumentsS2S():
    def __init__(self):
        self.batch_size = 8
        self.backward_freq = 16
        self.max_length = 1024
        self.print_freq = 100
        self.model_save_name = "bart_eli5_task_1_draft"
        self.learning_rate = 2e-4
        self.num_epochs = 1


s2s_args = ArgumentsS2S()

# LOAD TOKENIZER and MODEL S2S
qa_s2s_tokenizer, pre_model = make_qa_s2s_model(model_name="facebook/bart-base",
                                                from_file=None,
                                                device_modify="cpu")
qa_s2s_model = torch.nn.DataParallel(pre_model)

# LOAD and PREPROCESS DATA ELI5
s2s_train_dset, s2s_valid_dset = preprocess_data_eli5(dir_file="C:\ALL\OJT\TASK\data_eli5")

# TRAINING
train_qa_s2s(qa_s2s_model, qa_s2s_tokenizer, s2s_train_dset, s2s_valid_dset, s2s_args, device_modify="cpu")
