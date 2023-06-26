import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pickle

model_path = "C:/Users/Danila/BERT/model_2"
tokenizer_path = "C:/Users/Danila/BERT/tokenizer_2"
# Загрузка обученной модели и токенизатора
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

# Функция для классификации текста
def classify_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    outputs = model(input_ids, attention_mask)
    logits = outputs.logits
    predicted_labels = torch.argmax(logits, dim=1).tolist()

    return predicted_labels

#Открываем файлы для тестировки

with open('data/other/Names_Test1.plk','rb') as file1:
    fvlist = pickle.load(file1)
print(fvlist)
with open('data/other/Answers_Test1.pkl','rb') as file2:
    fvansw = pickle.load(file2)
print(fvansw)


texts = fvlist

# Классификация текстов
for text in texts:
    predicted_label = classify_text(text)
    print(f"Текст: '{text}' -> Предсказанная метка: {predicted_label}")

for i in range(100):
    txt = input('Введите овощ или фрукт')
    predicted_label = classify_text(txt)
    print(f"Текст: '{txt}' -> Предсказанная метка: {predicted_label}")