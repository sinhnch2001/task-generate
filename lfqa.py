import functools
import math
import torch
import os
import dotenv
import psycopg2
import pandas as pd

from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import AdamW, AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, \
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from nltk import PorterStemmer
from rouge import Rouge
from spacy.lang.en import English

pd.set_option("display.max_colwidth", None)


###############
# ELI5 seq2seq model training
###############


class ELI5DatasetS2S(Dataset):
    def __init__(self, examples_array, make_doc_fun=None, document_cache=None):
        self.data = examples_array
        self.make_doc_function = make_doc_fun
        self.document_cache = {} if document_cache is None else document_cache
        assert not (make_doc_fun is None and document_cache is None)

        self.qa_id_list = [(i, 0) for i in range(len(self.data))]

    def __len__(self):
        return len(self.qa_id_list)

    def make_example(self, idx):
        i, j = self.qa_id_list[idx]
        example = self.data[i]
        question = example["question"]
        answer = example["answers"][j]
        q_id = example["question_id"]
        if self.make_doc_function is not None:
            self.document_cache[q_id] = self.document_cache.get(
                q_id, self.make_doc_function(example["question"]))
        document = self.document_cache[q_id]
        in_st = "question: {} context: {}".format(
            question.lower().replace(" --t--", "").strip(), document.lower().strip())
        out_st = answer
        return in_st, out_st

    def __getitem__(self, idx):
        return self.make_example(idx)


def make_qa_s2s_model(model_name="facebook/bart-base", from_file=None, device_modify=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if device_modify == "cpu":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device_modify)
    if from_file is not None:
        # has model weights, optimizer, and scheduler states
        param_dict = torch.load(from_file)
        model.load_state_dict(param_dict["model"])
    return tokenizer, model


def make_qa_s2s_batch(qa_list, tokenizer, max_len=64, max_a_len=360, device=None):
    q_ls = [q for q, a in qa_list]
    a_ls = [a for q, a in qa_list]
    q_toks = tokenizer.batch_encode_plus(
        q_ls, max_length=max_len, pad_to_max_length=True)
    q_ids, q_mask = (torch.LongTensor(q_toks["input_ids"]).to(device),
                     torch.LongTensor(q_toks["attention_mask"]).to(device))
    a_toks = tokenizer.batch_encode_plus(a_ls, max_length=min(
        max_len, max_a_len), pad_to_max_length=True)
    a_ids, a_mask = (torch.LongTensor(a_toks["input_ids"]).to(device),
                     torch.LongTensor(a_toks["attention_mask"]).to(device))
    labels = a_ids[:, 1:].contiguous().clone()
    labels[a_mask[:, 1:].contiguous() == 0] = -100
    model_inputs = {"input_ids": q_ids,
                    "attention_mask": q_mask,
                    "decoder_input_ids": a_ids[:, :-1].contiguous(),
                    "labels": labels}
    return model_inputs


def train_qa_s2s_epoch(model, dataset, tokenizer, optimizer, scheduler, args, e=0, curriculum=False, device=None):
    model.train()

    # make iterator
    if curriculum:
        train_sampler = SequentialSampler(dataset)
    else:
        train_sampler = RandomSampler(dataset)

    model_collate_fn = functools.partial(
        make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device=device)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             sampler=train_sampler, collate_fn=model_collate_fn)
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)

    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()

    for step, batch_inputs in enumerate(epoch_iterator):
        outputs = model(**batch_inputs)
        loss = outputs.loss
        loss.backward()

        # optimizer
        if step % args.backward_freq == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        # some printing within the epoch
        loc_loss += loss.item()
        loc_steps += 1
        if step % args.print_freq == 0 or step == 1:
            print("{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                e, step, len(dataset) // args.batch_size, loc_loss / loc_steps, time() - st_time, ))
            loc_loss = 0
            loc_steps = 0


def eval_qa_s2s_epoch(model, dataset, tokenizer, args, device=None):
    model.eval()

    # make iterator
    train_sampler = SequentialSampler(dataset)
    model_collate_fn = functools.partial(make_qa_s2s_batch, tokenizer=tokenizer,
                                         max_len=args.max_length, device=device)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             sampler=train_sampler, collate_fn=model_collate_fn)
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)

    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    with torch.no_grad():
        for step, batch_inputs in enumerate(epoch_iterator):
            pre_loss = model(**batch_inputs)
            loss = pre_loss.loss
            loc_loss += loss.item()
            loc_steps += 1
            if step % args.print_freq == 0:
                print("{:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                    step, len(dataset) // args.batch_size, loc_loss / loc_steps, time() - st_time))
    print("Total \t L: {:.3f} \t -- {:.3f}".format(loc_loss /
                                                   loc_steps, time() - st_time))


def train_qa_s2s(qa_s2s_model, qa_s2s_tokenizer, s2s_train_dset, s2s_valid_dset, s2s_args, device_modify=None):
    s2s_optimizer = AdamW(qa_s2s_model.parameters(),
                          lr=s2s_args.learning_rate, eps=1e-8)
    s2s_scheduler = get_linear_schedule_with_warmup(s2s_optimizer, num_warmup_steps=400,
                                                    num_training_steps=(s2s_args.num_epochs + 1) * math.ceil(
                                                        len(s2s_train_dset) / s2s_args.batch_size), )
    for e in range(s2s_args.num_epochs):
        train_qa_s2s_epoch(qa_s2s_model, s2s_train_dset, qa_s2s_tokenizer, s2s_optimizer,
                           s2s_scheduler, s2s_args, e, curriculum=(e == 0), device=device_modify)
        m_save_dict = {
            "model": qa_s2s_model.module.state_dict() if hasattr(qa_s2s_model, 'module') else qa_s2s_model.state_dict(),
            "optimizer": s2s_optimizer.state_dict(),
            "scheduler": s2s_scheduler.state_dict()}

        print("Saving model {}".format(s2s_args.model_save_name))
        eval_qa_s2s_epoch(qa_s2s_model, s2s_valid_dset, qa_s2s_tokenizer, s2s_args, device=device_modify)
        torch.save(m_save_dict, "{}.pth".format(s2s_args.model_save_name))


# generate answer from input "question: ... context: <p> ..."
def qa_s2s_generate(question_doc, qa_s2s_model, qa_s2s_tokenizer, num_answers=1, num_beams=None,
                    min_len=64, max_len=256, do_sample=False, temp=1.0, top_p=None, top_k=None,
                    max_input_length=512, device_modify=None):
    model_inputs = make_qa_s2s_batch([(question_doc, "A")], qa_s2s_tokenizer,
                                     max_input_length, device=device_modify)

    n_beams = num_answers if num_beams is None else max(num_beams, num_answers)
    model = qa_s2s_model.module if hasattr(
        qa_s2s_model, 'module') else qa_s2s_model
    generated_ids = model.generate(input_ids=model_inputs["input_ids"],
                                   attention_mask=model_inputs["attention_mask"],
                                   min_length=min_len, max_length=max_len,
                                   do_sample=do_sample, early_stopping=True,
                                   num_beams=1 if do_sample else n_beams,
                                   temperature=temp, top_k=top_k, top_p=top_p,
                                   eos_token_id=qa_s2s_tokenizer.eos_token_id,
                                   no_repeat_ngram_size=3,
                                   num_return_sequences=num_answers,
                                   decoder_start_token_id=qa_s2s_tokenizer.bos_token_id)
    return [qa_s2s_tokenizer.decode(ans_ids, skip_special_tokens=True).strip() for ans_ids in generated_ids]


def compute_rouge_eli5(compare_list):
    stemmer = PorterStemmer()
    rouge = Rouge()
    nlpp = English()
    tokenizer = nlpp.tokenizer

    preds = [" ".join([stemmer.stem(str(w)) for w in tokenizer(pred)]) for gold, pred in compare_list]
    golds = [" ".join([stemmer.stem(str(w)) for w in tokenizer(gold)]) for gold, pred in compare_list]
    scores = rouge.get_scores(hyps=preds, refs=golds, avg=True)
    return scores


dotenv.load_dotenv()


def query_embd_gradient(embd, limit_doc=3, ):
    DBNAME = os.getenv("DBNAME", "wiki_db")
    HOST = os.getenv("HOST", "124.158.12.207")
    PORT = os.getenv("PORT", "15433")
    USER = os.getenv("USER", "gradlab")
    PWD = os.getenv("PASSWORD", "baldarg")
    TB_CLIENT = os.getenv("TB_CLIENT", "client_tb")
    TB_WIKI = os.getenv("TB_WIKI", "wiki_tb")
    MSD_WIKI = bool(os.getenv("MSD_WIKI", False))

    embd = str(list(embd.cpu().detach().numpy().reshape(-1)))
    try:
        connection = psycopg2.connect(dbname=DBNAME, host=HOST, port=PORT, user=USER, password=PWD)
        cursor = connection.cursor()
        aemb_sql = f"""
                        SET LOCAL ivfflat.probes = 3;
                        SELECT content 
                        FROM {TB_WIKI}
                        ORDER BY embedd <#> %s LIMIT %s;
                    """
        cursor.execute(aemb_sql, (embd, limit_doc))
        connection.commit()
        rows = cursor.fetchall()

        if connection:
            cursor.close()
            connection.close()

        return rows

    except (Exception, psycopg2.Error) as error:
        print("Failed query record from database {}".format(error))


import ast


def query_embd_mydatabase(embd, limit_doc=3, ):
    DBNAME = os.getenv("DBNAME", "wiki40b_snippets")
    HOST = os.getenv("HOST", "localhost")
    PORT = os.getenv("PORT", "5432")
    USER = os.getenv("USER", "postgres")
    PWD = os.getenv("PASSWORD", "12345")
    TB_WIKI = os.getenv("TB_WIKI", "wiki40")

    embd = str(list(embd.cpu().detach().numpy().reshape(-1)))
    try:
        connection = psycopg2.connect(dbname=DBNAME, host=HOST, port=PORT, user=USER, password=PWD)
        cursor = connection.cursor()
        aemb_sql = f"""
                                SET LOCAL ivfflat.probes = 3;
                                SELECT wiki_doc 
                                FROM {TB_WIKI}
                                ORDER BY wiki_emb <#> cube(%s) LIMIT %s;
                            """
        cursor.execute(aemb_sql, (ast.literal_eval(embd), limit_doc))
        connection.commit()
        rows = cursor.fetchall()

        if connection:
            cursor.close()
            connection.close()

        return rows

    except (Exception, psycopg2.Error) as error:
        print("Failed query record from database {}".format(error))


def insert_to_database_test():
    DBNAME = os.getenv("DBNAME", "kiki40")
    HOST = os.getenv("HOST", "")
    PORT = os.getenv("PORT", "5432")
    USER = os.getenv("USER", "postgres")
    PWD = os.getenv("PASSWORD", "12345")

    try:
        connection = psycopg2.connect(dbname=DBNAME, user=USER, host=HOST, port=PORT, password=PWD)
        cursor = connection.cursor()
        insert_doc = f"""
                        insert into wiki40(wiki_id, wiki_doc, wiki_emb) values(%s,%s,%s)
                    """
        values = ("gwerger", "dajsgdsajbcyuwiyqwuigqgdjahdjadjagd",
                  """0.7956, 0.0811, -0.3234, 0.6847, 0.1686, 0.2292, -0.6874, 0.5224,
                   0.5175, -1.3940, 0.0223, -0.5105, 0.0464, 0.0645, 0.0732, 0.0387,
                   0.9259, -0.2082, 0.6352, 0.0472, 0.2886, 0.6084, 0.2724, -0.2427,
                   0.6313, 0.1769, 0.0965, 0.0515, 0.1782, -0.6107, -0.4949, 1.0191,
                   -0.3270, 0.0507, -0.4845, 0.5949, -0.4574, -0.4514, 0.4395, 0.7720,
                   0.8380, -0.2629, 0.5365, 0.2077, 0.3900, 0.2899, 0.1023, 0.3890,
                   0.4701, -0.3880, 0.1439, -0.0152, 0.5325, -0.6031, 0.4036, 0.0428,
                   -1.0455, 0.7085, 0.0236, 0.5786, -1.1154, 1.0002, -0.3147, 0.3443,
                   0.6353, -0.2496, 0.4925, 0.1738, 0.5210, -0.3347, -1.1489, 0.7288,
                   0.1850, -1.1919, -0.6595, -0.1964, -0.4076, 0.3218, -0.1110, -0.7372,
                   0.7709, 0.4343, 0.0653, -0.8169, 0.3490, 0.4064, 0.1995, -0.0657,
                   0.0263, -0.4095, 0.0715, -0.3285, 0.5771, -0.3013, 0.2146, -0.0166,
                   0.0490, 0.4178, 0.0073, -0.6736, -1.0417, -1.0509, -0.2020, -0.3739,
                   0.4324, -0.1426, -0.6908, -0.7694, -0.0418, -0.9683, -0.3981, -0.2826,
                   -0.2165, -0.4561, -0.5563, -0.2243, 0.6288, 0.2617, 0.1422, -0.0401,
                   -0.5223, -0.9245, 0.6486, 0.5039, 0.3623, 0.0271, 0.1862, 0.9631""")
        cursor.execute(insert_doc, values)
        connection.commit()
        if connection:
            cursor.close()
            connection.close()

    except (Exception, psycopg2.Error) as error:
        print("Failed insert record from database {}".format(error))


def insert_to_database(wiki_id, wiki_doc, wiki_emb):
    DBNAME = os.getenv("DBNAME", "wiki40b_snippets")
    HOST = os.getenv("HOST", "localhost")
    PORT = os.getenv("PORT", "5432")
    USER = os.getenv("USER", "postgres")
    PWD = os.getenv("PASSWORD", "12345")

    try:
        connection = psycopg2.connect(dbname=DBNAME, user=USER, host=HOST, port=PORT, password=PWD)
        cursor = connection.cursor()
        insert_doc = f"""
                        insert into wiki40(wiki_id, wiki_doc, wiki_emb) values(%s,%s,%s)
                    """
        values = (wiki_id, wiki_doc, wiki_emb)
        cursor.execute(insert_doc, values)
        connection.commit()
        if connection:
            cursor.close()
            connection.close()

    except (Exception, psycopg2.Error) as error:
        print("Failed insert record from database: {}".format(error))


def load_model_qs(pretrain_name="vblagoje/dpr-question_encoder-single-lfqa-wiki", device=torch.device("cpu")):
    qs_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(pretrain_name)
    qs_model = DPRQuestionEncoder.from_pretrained(pretrain_name)
    qs_model.to(device)

    return qs_model, qs_tokenizer


def get_embds_qs(model, tokenizer, text, device):
    # Tokenize sentences
    model.eval()
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)

    return model_output['pooler_output']
