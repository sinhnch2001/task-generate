from lfqa import *
import json

# LOAD DATA VALIDATION
dir_file = "C:\ALL\OJT\TASK\eli5"
eli5_valid = json.load(open(dir_file + '\ELI5_val_10_doc.json'))
eli5_valid = eli5_valid[:2]

# LOAD TOKENIZER and MODEL S2S
qa_s2s_tokenizer, pre_model = make_qa_s2s_model(model_name="facebook/bart-base",
                                                from_file="bart_eli5_task_1_draft.pth",
                                                device_modify="cpu")
qa_s2s_model = torch.nn.DataParallel(pre_model)

predicted = []
reference = []

# Generate answers for the full test set
for i in range(len(eli5_valid)):
    # create support document with the dense index
    question = eli5_valid[i]['question']
    support_doc = "<P> " + " <P> ".join([str(p) for p in eli5_valid[i]["ctxs"]])
    # concatenate question and support document into BART input
    question_doc = "question: {} context: {}".format(question, support_doc)
    # generate an answer with beam search
    answer = qa_s2s_generate(question_doc, qa_s2s_model, qa_s2s_tokenizer,
                             num_answers=1, num_beams=8, min_len=96,
                             max_len=256, max_input_length=1024, device_modify="cpu")
    predicted += [answer[0]]
    reference += [eli5_valid[i]['answers'][0]]
    if i % 100 == 0: print("Step: ", i, "/", len(eli5_valid))

compare_list = [(g, p) for p, g in zip(predicted, reference)]
scores = compute_rouge_eli5(compare_list)
df = pd.DataFrame({
    'rouge1': [scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f']],
    'rouge2': [scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f']],
    'rougeL': [scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f']],
}, index=['P', 'R', 'F'])
print("RougeL for F1:", df["rougeL"][2])
