#%%
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
from datasets import load_metric
import torch



#%%

raw_datasets=load_dataset('csv', data_files={'train':'/home/ntan607/projects/BERT-IMDB/%50_Bottom_Fine_Tune/bottom50_train.csv','test': '/home/ntan607/projects/BERT-IMDB/%50_Bottom_Fine_Tune/bottom50_eval.csv'})
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


#%%
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]


model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)


training_args = TrainingArguments("test_trainer")


trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
)

trainer.train()



metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.evaluate()



model.save_pretrained("imdb_model_50_bottom_bert")
