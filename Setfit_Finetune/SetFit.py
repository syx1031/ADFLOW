import pandas as pd
# import torch.cuda
from datasets import Dataset
# from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def compute_metrics(y_pred, y_test):
    y_true_list = y_test
    y_pred_list = y_pred.numpy().tolist()
    accuracy = accuracy_score(y_true_list, y_pred_list)
    precision = precision_score(y_true_list, y_pred_list)
    recall = recall_score(y_true_list, y_pred_list)
    f1 = f1_score(y_true_list, y_pred_list)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


df = pd.read_csv('dataset_new.csv')
group = df.groupby('Label')

ad_df = group.get_group(True).reset_index(drop=True)
non_ad_df = group.get_group(False).reset_index(drop=True)
min_num = min(ad_df.shape[0], non_ad_df.shape[0], 10000) # 10000
ad_df = ad_df.sample(n=min_num, random_state=None)
non_ad_df = non_ad_df.sample(n=min_num, random_state=None)
df = pd.concat([ad_df, non_ad_df], axis=0, ignore_index=True).sample(frac=1).reset_index(drop=True).rename(columns={'Label': 'label', 'Text': 'text'})

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)

model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

argument = TrainingArguments(run_name="PG-OCR-test-3",
                             evaluation_strategy="steps",
                             eval_steps=5000,
                             # sampling_strategy="undersampling",
                             num_iterations=20,
                             save_strategy="steps",
                             save_steps=5000,
                             save_total_limit=100,
                             )
'''
argument = TrainingArguments(evaluation_strategy="steps",
                             eval_steps=1,
                             save_strategy="no",
                             batch_size=1)
'''

trainer = Trainer(model=model,
                  train_dataset=dataset["train"],
                  eval_dataset=dataset["test"],
                  args=argument,
                  metric=compute_metrics)

trainer.train()
metrics = trainer.evaluate()
print(metrics)
