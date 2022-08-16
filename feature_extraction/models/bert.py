import torch
import torch.nn as nn
from transformers import AutoModel


class Bert(nn.Module):

    def __init__(self, cfg):
        super(Bert, self).__init__()
        self.cfg = cfg
        #self.bert = AutoModel.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.bert =  AutoModel.from_pretrained(cfg.ENCODER) 
        # self.bert =  AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext-large") 
        #self.bert = AutoModel.from_pretrained('bert-base-chinese')

    def forward(self, x):
        # y = torch.ones((int(self.args.batch_size/4), self.args.max_textLen, self.args.textFea_dim),device=x.device)   
        y = self.bert(x, output_hidden_states=True, return_dict=True)
        # print(y.keys())
        # print(y.last_hidden_state)
        y = y.hidden_states
        y = [feature[:, 0, :] for feature in y[1:]]
        y = torch.cat(y)
        # print(len(y))
        # print(y.shape)
        return y
