
#%%
from typing import Sequence
import pandas as pd
import nltk
nltk.download('punkt')
data=pd.read_csv('/home/ntan607/projects/BERT-IMDB/FineSBERT/Data/SEvalBert.csv')
text = data['text']
text = list(text)
Sequences=[nltk.tokenize.sent_tokenize(i) for i in text]

#%%
import pickle
with open("/home/ntan607/projects/ListeRep.txt", "rb") as fp:   # Unpickling
     Liste = pickle.load(fp)


#%%
Index_Sent1=[]
Index_Sent2=[]
for i in range(len(Liste)):
    N_Liste1=[]
    N_Liste2=[]
    for j in range(len(Liste[i])):
        N_Liste1.append(Liste[i][j][1])
        N_Liste2.append(Liste[i][j][2])
    Index_Sent1.append(N_Liste1)
    Index_Sent2.append(N_Liste2)    

#%%
Red=[]
for x in range(len(Sequences)):
    N_Liste=zip(Index_Sent1[x],Index_Sent2[x])
    seq=[]
    for i,j in N_Liste:
        if len(Sequences[x][i]) < len(Sequences[x][j]):
            seq.append(Sequences[x][i])
        else:
            seq.append(Sequences[x][j])
    Red.append(list(set(seq)))

#%%
d={'text':[' '.join(Red[i]) for i in range(len(Red))], 'label':[label for label in data['label']], 'per':[100-100*len(' '.join(Red[i]))/len(text[i]) for i in range(len(Sequences))]}
df=pd.DataFrame.from_dict(d)
df.to_csv("ssberteval2.csv")


# %%
import pandas as pd
df=pd.read_csv('/home/ntan607/projects/Task2/sbertsimholey.csv')
# %%
Null_index=df[ (df['reduced'].isnull()) & (df['reduced']!='') ].index

# %%
update_df = df.drop(Null_index, inplace=True)
df.to_csv("newsbert.csv")
# %%
