#%% Dataset for BERT' filtration 

train='/home/ntan607/projects/Generating part/Data/neww/traingp.csv'
eval='/home/ntan607/projects/Generating part/Data/neww/evalgp.csv'

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
# %%
