import pandas as pd 
import torch 
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## load the data 
df = pd.read_csv("../data/mood_lyrics_cleaned.csv")

## keepin only neccessary cols 
df = df[["cleaned_lyrics","mood"]].dropna()

## encode labels 
le = LabelEncoder()
df["label"] = le.fit_transform(df["mood"])

## save the label mapping 
label_mapping = dict(zip(le.classes_,le.transform(le.classes_)))
print("Label Mapping:",label_mapping)

## tokenizer 
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

## tokenization function 
def tokenize(batch):
    return tokenizer(batch["cleaned_lyrics"],padding=True, truncation=True,max_length=512)

## convert to hugging face dataset 
dataset = Dataset.from_pandas(df[["cleaned_lyrics","label"]])
dataset = dataset.train_test_split(test_size=0.2,seed=42)
dataset = dataset.map(tokenize,batched=True)
dataset.set_format("torch",columns=["input_ids","attention_mask","label"])

## load the model 
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=len(label_mapping)).to(device)

## training arguments 
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",              # eval per epoch
    save_strategy="epoch",              # save per epoch ✅ now both match
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

##trainer 
trainer = Trainer(
    model = model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

#train 
trainer.train()

## save model and label map 
model.save_pretrained("../models/mood_classifier_bert")
tokenizer.save_pretrained("../models/mood_classifier_bert")
print("Model and tokenizer saved to: models/mood_classifier_bert")