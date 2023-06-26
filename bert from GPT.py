import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Загрузка предварительно обученной модели DistilBERT
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Определение меток классов
labels = ['фрукт', 'овощ']

# Функция классификации текста
def classify_text(text):
    # Токенизация текста
    tokens = tokenizer.encode_plus(text, truncation=True, padding=True, return_tensors='pt')
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    # Применение модели для классификации текста
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).squeeze()

    # Определение метки класса
    predicted_label_index = torch.argmax(probabilities).item()
    predicted_label = labels[predicted_label_index]

    return predicted_label, probabilities[predicted_label_index].item()

# Пример использования
text = "Это яблоко"
predicted_label, confidence = classify_text(text)
print(f"Текст: {text}")
print(f"Класс: {predicted_label}")
print(f"Уверенность: {confidence}")
