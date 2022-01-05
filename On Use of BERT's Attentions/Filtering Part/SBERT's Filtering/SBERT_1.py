#%%
from typing import Sequence
import pandas as pd
import nltk
nltk.download('punkt')
data=pd.read_csv('/home/ntan607/projects/BERT-IMDB/FineSBERT/Data/STrainBert.csv')
#%%
text = data['text']
text = list(text)
#%%
Sequences=[nltk.tokenize.sent_tokenize(i) for i in text]
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# %%
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

# Lsit of Similarity calculation of sentences in each sequence
#structure is [[[ ]]]
Liste=[]
for i,sentences in enumerate(Sequences):
    Liste.append(util.paraphrase_mining(model, sentences, corpus_chunk_size=len(sentences), top_k=1, max_pairs=100))
import pickle
with open("ListeReptr2.txt", "wb") as fp:   #Pickling
    pickle.dump(Liste, fp)

# %%
