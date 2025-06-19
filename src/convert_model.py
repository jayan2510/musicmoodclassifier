from transformers import BertForSequenceClassification

model_path = "../models/mood_classifier_bert"

model = BertForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
model.save_pretrained(model_path, safe_serialization=False)

print("✅ Model converted to pytorch_model.bin")
