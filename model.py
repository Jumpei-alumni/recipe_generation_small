from transformers import BartTokenizer, BartModel, BartConfig
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import random
import numpy as np
import torch.nn.functional as F

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

class RecipeBART(nn.Module):
    def __init__(self):
        super(RecipeBART, self).__init__()
        configuration = BartConfig(d_model=512, decoder_ffn_dim=1024, decoder_attention_heads=4, encoder_attention_heads=4, encoder_ffn_dim=1024, decoder_layers=3, encoder_layers=3)
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.bart = model = BartModel(configuration)
        self.dropout = nn.Dropout(0.1)
        self.projection_layer = nn.Linear(512, tokenizer.vocab_size)

    def forward(self, input, output, gpu, gene):
        inputs = input['input_ids']
        input_mask = input['attention_mask']
        if not gene:
            outputs = output['input_ids']
            output_mask = output['attention_mask']
        else:
            outputs = output
            output_mask = None
        if gpu:
            inputs = inputs.cuda()
            input_mask = input_mask.cuda()
            outputs = outputs.cuda()
            if output_mask is not None:
                output_mask = output_mask.cuda()
            
        bart_out = self.bart(
            input_ids=inputs,
            attention_mask=input_mask,
            decoder_input_ids=outputs,
            decoder_attention_mask=output_mask
            )[0]
        
        projection_input = self.dropout(bart_out)
        out = self.projection_layer(projection_input)
        return out #[b, L, V]

    def generate(self, name, ingr, gpu, tmp=0.9, top_k=40, top_p=0, max_len=200):
        tokenizer = self.tokenizer
        with torch.no_grad():
            input = tokenizer(name+'<'+ingr, return_tensors='pt')
            output = torch.LongTensor([[tokenizer.bos_token_id]])
            eos = torch.LongTensor([[tokenizer.eos_token_id]])
            if gpu:
                output = output.cuda()
                eos = eos.cuda()
            for i in range(max_len):
                word_p = self.forward(input, output, gpu, True)[0, -1].view(1, 1, -1)
                word = sampling_next_token_pk(tmp, word_p, top_k, top_p).view(1, 1)
                output = torch.cat([output, word], dim=-1)
                if word == eos:
                    break
        return output[0].cpu().numpy()


    def generate2(self, df, data_num, gpu, tmp=0.9, top_k=40, top_p=0, max_len=200):
        tokenizer = self.tokenizer
        data = get_data_from_df(df, data_num)
        name = data['name']
        ingr_num = np.random.randint(len(data['ingr']))
        input_ingr, output_ingr = get_ingredient(ingr_num, data['ingr'])
        generated = tokenizer.decode(self.generate(name, input_ingr, gpu, tmp, top_k, top_p, max_len))
        return {'name': name, 'ingr': input_ingr, 'gt':output_ingr+'>'+data['step'], 'ge':generated}


    def get_train_input(self, df, batch_size, max_length):
        tokenizer = self.tokenizer
        batch = np.random.randint(len(df), size=(batch_size, ))
        input_array = []
        output_array = []
        for data_num in batch:
            data = get_data_from_df(df, data_num)
            ingr_num = np.random.randint(len(data['ingr']))
            input_ingr, output_ingr = get_ingredient(ingr_num, data['ingr'])
            input_array.append(data['name']+'<'+input_ingr)
            output_array.append(output_ingr+'>'+data['step'])
        input = tokenizer(input_array, truncation=True, padding=True, return_tensors='pt')
        output = tokenizer(output_array, truncation=True, padding=True, return_tensors='pt', max_length=max_length)
        return {'input':input, 'output':output}

    def train_forward(self, df, batch_size, gpu, max_length):
        data = self.get_train_input(df, batch_size, max_length)
        model_output = self.forward(data['input'], data['output'], gpu, False)
        output = data['output']['input_ids']
        if gpu:
            output = output.cuda()
        return model_output, output



def get_ingredient(n, ingrs):
    if n >= len(ingrs):
        n = len(ingrs)
    random.shuffle(ingrs)
    return ",".join(ingrs[:n]), ",".join(ingrs)

def get_data_from_df(df, data_num):
    name = df['name'][data_num]
    ingr = df['ingredients'][data_num].split(',')
    step = df['steps'][data_num]  
    return {'name':name, 'ingr':ingr, 'step':step}

import torch.nn.functional as F
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    
    
    return logits

def sampling_next_token_pk(temperature, logits, top_k=0, top_p=0.0):


  # Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
  logits = logits[0, -1, :] / temperature
  filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

  # Sample from the filtered distribution
  probabilities = F.softmax(filtered_logits, dim=-1)
  
  next_token = torch.multinomial(probabilities, 1)

  return next_token