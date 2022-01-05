
#%%

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining
from torch.utils.data import Dataset      
import random               
import torch
import os


DEBUG           = False

INPUT_DIR       = 'articles'

USE_APEX        = True
APEX_OPT_LEVEL  = 'O1'

MODEL           = 'gpt2-medium' #{gpt2, gpt2-medium, gpt2-large, gpt2-xl}

UNFREEZE_LAST_N = 6 #The last N layers to unfreeze for training

SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}
                    
MAXLEN          = 1024  #{768, 1024, 1280, 1600}

TRAIN_SIZE      = 0.8

if USE_APEX:
    TRAIN_BATCHSIZE = 4
    BATCH_UPDATE    = 16
else:
    TRAIN_BATCHSIZE = 2
    BATCH_UPDATE    = 32

EPOCHS          = 4
LR              = 5e-4
EPS             = 1e-8
WARMUP_STEPS    = 1e2

SEED            = 2020

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

class myDataset(Dataset):

    def __init__(self, data, tokenizer, randomize=True):

        sentiment,text, keywords = [], [], []
        for k, v in data.items():
            sentiment.append(v[0])
            text.append(v[1])
            keywords.append(v[2])

        self.randomize = randomize
        self.tokenizer = tokenizer 
        self.sentiment = sentiment
        self.text      = text
        self.keywords  = keywords  

    #---------------------------------------------#

    @staticmethod
    def join_keywords(keywords, randomize=True):
        N = len(keywords)

        #random sampling and shuffle
        if randomize: 
            M = random.choice(range(N+1))
            keywords = keywords[:M]
            random.shuffle(keywords)

        return ','.join(keywords)

    #---------------------------------------------#

    def __len__(self):
        return len(self.text)

    #---------------------------------------------#
    
    def __getitem__(self, i):
        keywords = self.keywords[i].copy()
        kw = self.join_keywords(keywords, self.randomize)
        
        input = SPECIAL_TOKENS['bos_token'] + self.sentiment[i] +\
                SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS['sep_token'] + \
                self.text[i] + SPECIAL_TOKENS['eos_token']

        encodings_dict = tokenizer(input,                                   
                                   truncation=True, 
                                   max_length=MAXLEN, 
                                   padding="max_length")   
        
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']
        
        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids), 
                'attention_mask': torch.tensor(attention_mask)}

def get_tokenier(special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL) #GPT2Tokenizer

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    return tokenizer

def get_model(tokenizer, special_tokens=None, load_model_path=None):

    #GPT2LMHeadModel
    if special_tokens:
        config = AutoConfig.from_pretrained(MODEL, 
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else: 
        config = AutoConfig.from_pretrained(MODEL,                                     
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)    

    #----------------------------------------------------------------#
    model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)

    if special_tokens:
        #Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    model.cuda()
    return model



#%%
tokenizer = get_tokenier(special_tokens=SPECIAL_TOKENS)
model = get_model(tokenizer, 
                  special_tokens=SPECIAL_TOKENS,
                  load_model_path='/home/ntan607/projects/Gen_Part/newgpt2/pytorch_model.bin')






#%Dataset prep
#%% Dataset for BERT' filtration 

train='/home/ntan607/projects/Gen_Part/traingp.csv'
eval='/home/ntan607/projects/Gen_Part/evalgp.csv'

#%% Arranging %6 dataset
import pandas as pd
dfeval=pd.read_csv(eval)
dftrain=pd.read_csv(train)
dftrain=dftrain[["100", "6", "label"]]
dfeval=dfeval[["100", "6", "label"]]


#%% Separating 40000/10000 data point with equal positive and negative sentiments.
pos=dftrain.iloc[0:12501].append(dfeval.iloc[0:7499])
neg=dftrain.iloc[12501:25001].append(dfeval.iloc[17499:25001])
newtrain=pos.append(neg)
neweval=dfeval.iloc[7499:17499]

#%% Coverting numeric labels to string
Sentiment_train=[]
for i in newtrain['label']:
    if i==1:
        Sentiment_train.append("positive")
    else:
        Sentiment_train.append("negative")
Sentiment_eval=[]   
for i in neweval['label']:
    if i==1:
        Sentiment_eval.append("positive")
    else:
        Sentiment_eval.append("negative")

# %%
train__6=[]
eval__6=[]
train_data={}
eval_data={}
for i in range(len(newtrain)):
    train__6.append([Sentiment_train[i],newtrain['100'].values.tolist()[i],newtrain['6'].values.tolist()[i].split()])
    train_data[i]=train__6[i]
for i in range(len(neweval)):
    eval__6.append([Sentiment_eval[i],neweval['100'].values.tolist()[i],neweval['6'].values.tolist()[i].split()])
    eval_data[i]=eval__6[i]

#%%
Gen_list=[]
sentiment=''
for keywords in [eval_data[0][2]]:
#[['practice','encouraging', 'international', 'study','students','beneficial','country' ]]:
    kw = myDataset.join_keywords(keywords, randomize=False)
    prompt = SPECIAL_TOKENS['bos_token'] + sentiment + \
             SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS['sep_token']
         
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    device = torch.device("cuda")
    generated = generated.to(device)

    model.eval();
    sample_outputs = model.generate(generated, 
                                do_sample=True,   
                                max_length=1000, 
                                min_length=100,                                                      
                                num_beams=3,
                                repetition_penalty=3.0,
                                early_stopping=True,      
                                num_return_sequences=1
                                )

    for i, sample_output in enumerate(sample_outputs):
        text = tokenizer.decode(sample_output, skip_special_tokens=True)
        a = len(sentiment) + len(','.join(keywords))    
        print("{}: {}\n\n".format(i+1,  text[a:]))
    Gen_list.append(text[a:])












#%%
import pandas as pd
dfeval=pd.read_csv("/home/ntan607/projects/GPT2-GEN/Data/eval.csv",index_col=0)
neweval=dfeval.iloc[7499:17499,0:5]
Gen_data=[]
for i in range(len(neweval)):
    Gen_data.append(neweval['3'].values.tolist()[i].split())
#%%
Gen_list=[]
sentiment=''
for keywords in Gen_data:
    kw = myDataset.join_keywords(keywords, randomize=False)
    prompt = SPECIAL_TOKENS['bos_token'] + sentiment + \
             SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS['sep_token']
         
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    device = torch.device("cuda")
    generated = generated.to(device)

    model.eval();
    sample_outputs = model.generate(generated, 
                                do_sample=True,   
                                max_length=520, 
                                min_length=100,                                                      
                                top_k=30,
                                top_p=0.7,
                                temperature=0.9,
                                repetition_penalty=3.0,
                                early_stopping=True,      
                                num_return_sequences=1
                                )

    for i, sample_output in enumerate(sample_outputs):
        text = tokenizer.decode(sample_output, skip_special_tokens=True)
        a = len(sentiment) + len(','.join(keywords))    
        print("{}: {}\n\n".format(i+1,  text[a:]))
    Gen_list.append(text[a:])
df = pd.DataFrame (Gen_list, columns = ['text'])
#df.to_csv("gen_top_6.csv")
