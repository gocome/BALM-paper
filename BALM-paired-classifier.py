from datetime import date
import numpy as np
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaForMaskedLM,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import warnings
warnings.simplefilter('ignore')


# Load your pretrained MLM checkpoint
pretrained_model_name = f"../models/BALM-paired_LC-coherence_90-5-5-split_122222"
mlm_model = RobertaForMaskedLM.from_pretrained(pretrained_model_name)

# Get its config
config = mlm_model.config
# Modify config for classification
config.num_labels = 2   # set to your number of classes

# Initialize classification model with same config
cls_model = RobertaForSequenceClassification(config)
# Transfer encoder weights
cls_model.roberta.load_state_dict(mlm_model.roberta.state_dict())

# Freeze all encoder params
for param in cls_model.roberta.parameters():
    param.requires_grad = False
# (Optional) unfreeze last encoder layer for better adaptation
# for param in cls_model.roberta.encoder.layer[-1].parameters():
#     param.requires_grad = True

tokenizer = RobertaTokenizer.from_pretrained("tokenizer")

# Define tokenize function
def tokenize_function(x):
    # Split text and labels
    texts = []
    labels = []
    for line in x["text"]:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            texts.append(parts[0])
            labels.append(int(parts[1]))
    
    # Tokenize the texts
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True,
    )
    
    # Add labels
    tokenized["labels"] = labels
    return tokenized

# load the tran, eval, and test data
data_files = {
    "train": ['./data/train-test-eval_paired/train_labelled.txt'],
    "eval": ['./data/train-test-eval_paired/eval_labelled.txt'],
    "test": ['./data/train-test-eval_paired/test_labelled.txt']
}
dataset = load_dataset("text", data_files=data_files)
tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=["text"],
    batched=True,
    desc="Running tokenizer on dataset",
)

# Update training arguments to include label column
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-4,              # slightly higher since only classifier is learning
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    num_train_epochs=30,              # may need more epochs since fewer params are trainable
    weight_decay=0.01,
    warmup_steps=25000,
    logging_dir="./logs",
    logging_steps=10000,
    label_names=["labels"],
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}


trainer = Trainer(
    model=cls_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    compute_metrics=compute_metrics,
)

trainer.train()

test_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
print(test_results)

trainer.save_model(f"../models/BALM-paired-classifier_LC-coherence_90-5-5-split_{date.today().isoformat()}")

print("Training complete.")
