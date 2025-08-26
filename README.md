# Week 5: Advanced NLP & Cyberbullying Detection

## Objective
Develop a robust **Cyberbullying Detection Model** using advanced NLP techniques, including:

- Feature extraction
- Sentiment analysis
- Transformer-based classification (DistilBERT)
- Model evaluation with confusion matrix, ROC curve, and precision-recall curve
- Visualization of misclassified tweets using a Word Cloud

---

## Dataset
- File: `cyberbullying_dataset.csv`  
- Columns:
  - `text`: Tweet text
  - `label`: Cyberbullying type
- Task: Binary classification  
  - `0` → Not Cyberbullying
  - `1` → Cyberbullying (any type)

---

## Step 1: Load Libraries and Dataset
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.class_weight import compute_class_weight
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch.nn as nn

df = pd.read_csv("https://raw.githubusercontent.com/ramizcihe/week5-cihe240058/refs/heads/main/cyberbullying_dataset.csv")
df.dropna(subset=["text"], inplace=True)
df["binary_label"] = df["label"].apply(lambda x: 0 if x == "not_cyberbullying" else 1)


Step 2: Feature Engineering

Sentiment Analysis using TextBlob

N-gram Features (for visualization only)

code:
df["sentiment"] = df["text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
vectorizer = CountVectorizer(ngram_range=(1,2), max_features=100)
ngram_matrix = vectorizer.fit_transform(df["text"])


Step 3: Dataset Preparation for BERT

Use DistilBERT tokenizer

Create PyTorch Dataset wrapper

from torch.utils.data import Dataset

class CyberDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
final_train_ds = CyberDataset(df["text"].tolist(), df["binary_label"].tolist(), tokenizer)

Step 4: Compute Class Weights
cw = compute_class_weight(class_weight="balanced",
                          classes=np.unique(df["binary_label"].values),
                          y=df["binary_label"].values)
class_weights_tensor = torch.tensor(cw, dtype=torch.float32)

Step 5: Custom Weighted Trainer
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

Step 6: Training the Model
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import accuracy_score, f1_score
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=final_train_ds,
    eval_dataset=final_train_ds,
    compute_metrics=compute_metrics,
    class_weights=class_weights_tensor
)

trainer.train()

Step 7: Evaluation & Visualization
Confusion Matrix
from sklearn.metrics import confusion_matrix
preds_output = trainer.predict(final_train_ds)
preds = np.argmax(preds_output.predictions, axis=1)
labels_true = df["binary_label"].values

cm = confusion_matrix(labels_true, preds)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.show()

ROC Curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(labels_true, preds_output.predictions[:,1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0,1],[0,1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

Precision-Recall Curve
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(labels_true, preds_output.predictions[:,1])
plt.plot(recall, precision, lw=2, color="purple")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

Word Cloud of Misclassified Tweets
misclassified_texts = df["text"][labels_true != preds]
if len(misclassified_texts) > 0:
    text_combined = " ".join(misclassified_texts)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_combined)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Misclassified Tweets")
    plt.show()
else:
    print("✅ No misclassified tweets found.")

✅ Summary

Successfully loaded and preprocessed the dataset

Added sentiment polarity and n-gram features

Trained a weighted DistilBERT model for binary classification

Evaluated using confusion matrix, ROC, PR curves

Visualized misclassified tweets using Word Cloud
