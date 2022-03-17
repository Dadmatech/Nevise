import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import transformers

#################################################
# SubwordBert
#################################################


class SubwordBert(nn.Module):
    def __init__(self, screp_dim, padding_idx, output_dim):
        super(SubwordBert,self).__init__()

        self.bert_dropout = torch.nn.Dropout(0.2)
        self.bert_model = transformers.BertModel.from_pretrained("HooshvareLab/bert-fa-base-uncased")
        self.bertmodule_outdim = self.bert_model.config.hidden_size
        # Uncomment to freeze BERT layers
        # for param in self.bert_model.parameters():
        #     param.requires_grad = False

        # output module
        assert output_dim>0
        # self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.bertmodule_outdim,output_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean',ignore_index=padding_idx)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_merged_encodings(self, bert_seq_encodings, seq_splits, mode='avg'):
        bert_seq_encodings = bert_seq_encodings[:sum(seq_splits)+2,:] # 2 for [CLS] and [SEP]
        bert_seq_encodings = bert_seq_encodings[1:-1,:]
        # a tuple of tensors
        split_encoding = torch.split(bert_seq_encodings,seq_splits,dim=0)
        batched_encodings = pad_sequence(split_encoding,batch_first=True,padding_value=0)
        if mode=='avg':
            seq_splits = torch.tensor(seq_splits).reshape(-1,1).to(self.device)
            out = torch.div( torch.sum(batched_encodings,dim=1), seq_splits)
        elif mode=="add":
            out = torch.sum(batched_encodings,dim=1)
        else:
            raise Exception("Not Implemented")
        return out

    def forward(self,
                batch_bert_dict: "{'input_ids':tensor, 'attention_mask':tensor, 'token_type_ids':tensor}",
                batch_splits: "list[list[int]]",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None,
                topk = 1):
        
        # cnn
        batch_size = len(batch_splits)

        # bert
        # BS X max_nsubwords x self.bertmodule_outdim
        bert_encodings, cls_encoding = self.bert_model(
            input_ids=batch_bert_dict["input_ids"],
            attention_mask=batch_bert_dict["attention_mask"],
            token_type_ids=batch_bert_dict["token_type_ids"],
            return_dict=False
        )
        bert_encodings = self.bert_dropout(bert_encodings)
        # BS X max_nwords x self.bertmodule_outdim
        bert_merged_encodings = pad_sequence(
            [self.get_merged_encodings(bert_seq_encodings, seq_splits, mode='avg') \
                for bert_seq_encodings, seq_splits in zip(bert_encodings,batch_splits)],
            batch_first=True,
            padding_value=0
        )

        # concat aux_embs
        # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
        intermediate_encodings = bert_merged_encodings
        if aux_word_embs is not None:
            intermediate_encodings = torch.cat((intermediate_encodings,aux_word_embs),dim=2)

        # dense
        # [BS,max_nwords,*] or [BS,max_nwords,self.bertmodule_outdim]->[BS,max_nwords,output_dim]
        # logits = self.dense(self.dropout(intermediate_encodings))
        logits = self.dense(intermediate_encodings)

        # loss
        if targets is not None:
            assert len(targets)==batch_size                # targets:[[BS,max_nwords]
            logits_permuted = logits.permute(0, 2, 1)               # logits: [BS,output_dim,max_nwords]
            loss = self.criterion(logits_permuted,targets)
        
        # eval preds
        if not self.training:
            probs = F.softmax(logits,dim=-1)            # [BS,max_nwords,output_dim]
            if topk>1:
                topk_values, topk_inds = \
                    torch.topk(probs, topk, dim=-1, largest=True, sorted=True)  # -> (Tensor, LongTensor) of [BS,max_nwords,topk]
            elif topk==1:
                topk_inds = torch.argmax(probs,dim=-1)   # [BS,max_nwords]

            # Note that for those positions with padded_idx,
            #   the arg_max_prob above computes a index because 
            #   the bias term leads to non-uniform values in those positions
            
            return loss.cpu().detach().numpy(), topk_inds.cpu().detach().numpy()
        return loss

