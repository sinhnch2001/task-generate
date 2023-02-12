from lfqa import *

# LOAD TOKENIZER and MODEL S2S
qa_s2s_tokenizer, pre_model = make_qa_s2s_model(model_name="facebook/bart-base",
                                                from_file="bart_eli5_task_1_draft.pth",
                                                device_modify="cpu")
qa_s2s_model = torch.nn.DataParallel(pre_model)

device = torch.device('cpu')  # Change this if you want


def retrieve(question, qs_model, qs_tokenizer, device, limit_doc=1):
    question_embd = get_embds_qs(qs_model, qs_tokenizer, question, device=device)
    documents_wiki = query_embd_mydatabase(question_embd, limit_doc=limit_doc)
    return [doc[-1] for doc in documents_wiki]


qs_model, qs_tokenizer = load_model_qs(device=device)

while 1:
    question = input("\nUSER: ")
    if question == "[EXIT]":
        break
    else:
        doc_5 = retrieve(question, qs_model, qs_tokenizer, device, limit_doc=1)
        doc = "<P> " + " <P> ".join([p for p in doc_5])
        question_doc = "question: {} context: {}".format(question, doc)

        # generate an answer with beam search
        answer = qa_s2s_generate(
            question_doc, qa_s2s_model, qa_s2s_tokenizer,
            num_answers=1,
            num_beams=8,
            min_len=3,
            max_len=100,
            max_input_length=1024,
            device_modify="cpu")[0]

        print("\nBOT:", answer.replace("\n", ""))