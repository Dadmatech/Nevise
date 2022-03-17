from tqdm import tqdm
import os, sys
import numpy as np
import pickle
import numpy as np
import transformers
import torch
from torch.nn.utils.rnn import pad_sequence

def progressBar(value, endvalue, names, values, bar_length=30):
    assert(len(names)==len(values));
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow));
    string = '';
    for name, val in zip(names,values):
        temp = '|| {0}: {1:.4f} '.format(name, val) if val!=None else '|| {0}: {1} '.format(name, None)
        string+=temp;
    sys.stdout.write("\rPercent: [{0}] {1}% {2}".format(arrow + spaces, int(round(percent * 100)), string))
    sys.stdout.flush()
    return

def load_data(base_path, corr_file, incorr_file):
    
    # load files
    if base_path:
        assert os.path.exists(base_path)==True
    incorr_data = []
    opfile1 = open(os.path.join(base_path, incorr_file),"r")
    for line in opfile1:
        if line.strip()!="": incorr_data.append(line.strip())
    opfile1.close()
    corr_data = []
    opfile2 = open(os.path.join(base_path, corr_file),"r")
    for line in opfile2:
        if line.strip()!="": corr_data.append(line.strip())
    opfile2.close()
    assert len(incorr_data)==len(corr_data)
    
    # verify if token split is same
    for i,(x,y) in tqdm(enumerate(zip(corr_data,incorr_data))):
        x_split, y_split = x.split(), y.split()
        try:
            assert len(x_split)==len(y_split)
        except AssertionError:
            print("# tokens in corr and incorr mismatch. retaining and trimming to min len.")
            # print(x_split, y_split)
            # mn = min([len(x_split),len(y_split)])
            # corr_data[i] = " ".join(x_split[:mn])
            # incorr_data[i] = " ".join(y_split[:mn])
            # print(corr_data[i],incorr_data[i])
    
    # return as pairs
    data = []
    for x,y in tqdm(zip(corr_data,incorr_data)):
        data.append((x,y))
    
    print(f"loaded tuples of (corr,incorr) examples from {base_path}")
    return data



def batch_iter(data, batch_size, shuffle):
    """
    each data item is a tuple of lables and text
    """
    n_batches = int(np.ceil(len(data) / batch_size))
    indices = list(range(len(data)))
    if shuffle:  np.random.shuffle(indices)

    for i in range(n_batches):
        batch_indices = indices[i * batch_size: (i + 1) * batch_size]
        batch_labels = [data[idx][0] for idx in batch_indices]
        batch_sentences = [data[idx][1] for idx in batch_indices]
       
        yield (batch_labels,batch_sentences)

def labelize(batch_labels, vocab):
    token2idx, pad_token, unk_token = vocab["token2idx"], vocab["pad_token"], vocab["unk_token"]
    list_list = [[token2idx[token] if token in token2idx else token2idx[unk_token] for token in line.split()] for line in batch_labels]
    list_tensors = [torch.tensor(x) for x in list_list]
    tensor_ = pad_sequence(list_tensors,batch_first=True,padding_value=token2idx[pad_token])
    return tensor_, torch.tensor([len(x) for x in list_list]).long()

def tokenize(batch_sentences, vocab):
    token2idx, pad_token, unk_token = vocab["token2idx"], vocab["pad_token"], vocab["unk_token"]
    list_list = [[token2idx[token] if token in token2idx else token2idx[unk_token] for token in line.split()] for line in batch_sentences]
    list_tensors = [torch.tensor(x) for x in list_list]
    tensor_ = pad_sequence(list_tensors,batch_first=True,padding_value=token2idx[pad_token])
    return tensor_, torch.tensor([len(x) for x in list_list]).long()


def untokenize(batch_predictions, batch_lengths, vocab):
    idx2token = vocab["idx2token"]
    unktoken = vocab["unk_token"]
    assert len(batch_predictions)==len(batch_lengths)
    batch_predictions = \
        [ " ".join( [idx2token[idx] for idx in pred_[:len_]] ) \
         for pred_,len_ in zip(batch_predictions,batch_lengths) ]
    return batch_predictions

def untokenize_without_unks(batch_predictions, batch_lengths, vocab, batch_clean_sentences, backoff="pass-through"):
    assert backoff in ["neutral","pass-through"], print(f"selected backoff strategy not implemented: {backoff}")
    idx2token = vocab["idx2token"]
    unktoken = vocab["token2idx"][vocab["unk_token"]]
    assert len(batch_predictions)==len(batch_lengths)==len(batch_clean_sentences)
    batch_clean_sentences = [sent.split() for sent in batch_clean_sentences]
    if backoff=="pass-through":
        batch_predictions = \
            [ " ".join( [   idx2token[idx] if idx!=unktoken else clean_[i] for i, idx in enumerate(pred_[:len_])  ] ) \
             for pred_,len_,clean_ in zip(batch_predictions,batch_lengths,batch_clean_sentences) ]
    elif backoff=="neutral":
        batch_predictions = \
            [ " ".join( [   idx2token[idx] if idx!=unktoken else "a" for i, idx in enumerate(pred_[:len_])  ] ) \
             for pred_,len_,clean_ in zip(batch_predictions,batch_lengths,batch_clean_sentences) ]
    return batch_predictions

def untokenize_without_unks2(batch_predictions, batch_lengths, vocab, batch_clean_sentences, topk=None):
    """
    batch_predictions are softmax probabilities and should have shape (batch_size,max_seq_len,vocab_size)
    batch_lengths should have shape (batch_size)
    batch_clean_sentences should be strings of shape (batch_size)
    """
    #print(batch_predictions.shape)
    idx2token = vocab["idx2token"]
    unktoken = vocab["token2idx"][vocab["unk_token"]]
    assert len(batch_predictions)==len(batch_lengths)==len(batch_clean_sentences)
    batch_clean_sentences = [sent.split() for sent in batch_clean_sentences]

    if topk is not None:
        # get topk items from dim=2 i.e top 5 prob inds
        batch_predictions = np.argpartition(-batch_predictions,topk,axis=-1)[:,:,:topk] # (batch_size,max_seq_len,5)
    #else:
    #    batch_predictions = batch_predictions # already have the topk indices

    # get topk words
    idx_to_token = lambda idx,idx2token,corresponding_clean_token,unktoken: idx2token[idx] if idx!=unktoken else corresponding_clean_token
    batch_predictions = \
    [[[idx_to_token(wordidx,idx2token,batch_clean_sentences[i][j],unktoken) \
       for wordidx in topk_wordidxs] \
      for j,topk_wordidxs in enumerate(predictions[:batch_lengths[i]])]  \
     for i,predictions in enumerate(batch_predictions)]

    return batch_predictions



def get_model_nparams(model):
    ntotal = 0
    for param in list(model.parameters()):
        temp = 1
        for sz in list(param.size()): temp*=sz
        ntotal += temp
    return ntotal


def load_vocab_dict(path_: str):
    """
    path_: path where the vocab pickle file is saved
    """
    with open(path_, 'rb') as fp:
        vocab = pickle.load(fp)
    return vocab



BERT_TOKENIZER = transformers.BertTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased", do_lower_case=False)
BERT_TOKENIZER.do_basic_tokenize = False
BERT_TOKENIZER.tokenize_chinese_chars = False
BERT_MAX_SEQ_LEN = 512

def merge_subtokens(tokens: "list"):
    merged_tokens = []
    for token in tokens:
        if token.startswith("##"): merged_tokens[-1] = merged_tokens[-1]+token[2:]
        else: merged_tokens.append(token)
    text = " ".join(merged_tokens)
    return text


def _custom_bert_tokenize_sentence(text):
    # from hazm import WordTokenizer
    new_tokens = []
    tokens = BERT_TOKENIZER.tokenize(text)
    j = 0
    for i, t in enumerate(tokens):
        if t == '[UNK]':
            new_tokens.append(text.split()[j])
        else:
            new_tokens.append(t)
        if t[0] != '#':
            j += 1
    tokens = new_tokens
    tokens = tokens[:BERT_MAX_SEQ_LEN-2] # 2 allowed for [CLS] and [SEP]
    idxs = np.array([idx for idx,token in enumerate(tokens) if not token.startswith("##")]+[len(tokens)])
    split_sizes = (idxs[1:]-idxs[0:-1]).tolist()
    # NOTE: BERT tokenizer does more than just splitting at whitespace and tokenizing. So be careful.
    # -----> assert len(split_sizes)==len(text.split()), print(len(tokens), len(split_sizes), len(text.split()), split_sizes, text)
    # -----> hence do the following:
    text = merge_subtokens(tokens)
    assert len(split_sizes)==len(text.split()), print(len(tokens), len(split_sizes), len(text.split()), split_sizes, text)
    return text, tokens, split_sizes


def _custom_bert_tokenize_sentences(list_of_texts):
    out = [_custom_bert_tokenize_sentence(text) for text in list_of_texts]
    texts, tokens, split_sizes = list(zip(*out))
    return [*texts], [*tokens], [*split_sizes]

_simple_bert_tokenize_sentences = \
    lambda list_of_texts: [merge_subtokens( BERT_TOKENIZER.tokenize(text)[:BERT_MAX_SEQ_LEN-2] ) for text in list_of_texts]


def bert_tokenize(batch_sentences):
    """
    inputs:
        batch_sentences: List[str]
            a list of textual sentences to tokenized
    outputs:
        batch_attention_masks, batch_input_ids, batch_token_type_ids
            2d tensors of shape (bs,max_len)
        batch_splits: List[List[Int]]
            specifies #sub-tokens for each word in each textual string after sub-word tokenization
    """
    batch_sentences, batch_tokens, batch_splits = _custom_bert_tokenize_sentences(batch_sentences)
    
    # max_seq_len = max([len(tokens) for tokens in batch_tokens])
    # batch_encoded_dicts = [BERT_TOKENIZER.encode_plus(tokens,max_length=max_seq_len,pad_to_max_length=True) for tokens in batch_tokens]
    batch_encoded_dicts = [BERT_TOKENIZER.encode_plus(tokens) for tokens in batch_tokens]

    batch_attention_masks = pad_sequence([torch.tensor(encoded_dict["attention_mask"]) for encoded_dict in batch_encoded_dicts],batch_first=True,padding_value=0)
    batch_input_ids = pad_sequence([torch.tensor(encoded_dict["input_ids"]) for encoded_dict in batch_encoded_dicts],batch_first=True,padding_value=0)
    batch_token_type_ids = pad_sequence([torch.tensor(encoded_dict["token_type_ids"]) for encoded_dict in batch_encoded_dicts],batch_first=True,padding_value=0)

    batch_bert_dict = {"attention_mask":batch_attention_masks,
                       "input_ids":batch_input_ids,
                       "token_type_ids":batch_token_type_ids}

    return batch_sentences, batch_bert_dict, batch_splits


def bert_tokenize_for_valid_examples(batch_orginal_sentences, batch_noisy_sentences):
    """
    inputs:
        batch_noisy_sentences: List[str]
            a list of textual sentences to tokenized
        batch_orginal_sentences: List[str]
            a list of texts to make sure lengths of input and output are same in the seq-modeling task
    outputs (only of batch_noisy_sentences):
        batch_attention_masks, batch_input_ids, batch_token_type_ids
            2d tensors of shape (bs,max_len)
        batch_splits: List[List[Int]]
            specifies #sub-tokens for each word in each textual string after sub-word tokenization
    """
    _batch_orginal_sentences = _simple_bert_tokenize_sentences(batch_orginal_sentences)
    _batch_noisy_sentences, _batch_tokens, _batch_splits = _custom_bert_tokenize_sentences(batch_noisy_sentences)
    valid_idxs = [idx for idx,(a,b) in enumerate(zip(_batch_orginal_sentences, _batch_noisy_sentences)) if len(a.split())==len(b.split())]
    batch_orginal_sentences = [line for idx,line in enumerate(_batch_orginal_sentences) if idx in valid_idxs]
    batch_noisy_sentences = [line for idx,line in enumerate(_batch_noisy_sentences) if idx in valid_idxs]
    batch_tokens = [line for idx,line in enumerate(_batch_tokens) if idx in valid_idxs]
    batch_splits = [line for idx,line in enumerate(_batch_splits) if idx in valid_idxs]
    
    batch_bert_dict = {"attention_mask":[],"input_ids":[],"token_type_ids":[]}
    if len(valid_idxs)>0:
        batch_encoded_dicts = [BERT_TOKENIZER.encode_plus(tokens) for tokens in batch_tokens]
        batch_attention_masks = pad_sequence([torch.tensor(encoded_dict["attention_mask"]) for encoded_dict in batch_encoded_dicts],batch_first=True,padding_value=0)
        batch_input_ids = pad_sequence([torch.tensor(encoded_dict["input_ids"]) for encoded_dict in batch_encoded_dicts],batch_first=True,padding_value=0)
        batch_token_type_ids = pad_sequence([torch.tensor(encoded_dict["token_type_ids"]) for encoded_dict in batch_encoded_dicts],batch_first=True,padding_value=0)
        batch_bert_dict = {"attention_mask":batch_attention_masks, 
                           "input_ids":batch_input_ids,
                           "token_type_ids":batch_token_type_ids}

    return batch_orginal_sentences, batch_noisy_sentences, batch_bert_dict, batch_splits