import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from torch.utils.data import DataLoader, random_split
from transformers import AdamW
import pickle
from tqdm import tqdm

# Загрузка тренировочных данных
with open('C:/Users/Danila/BERT/data/Train_pkl/training_list.pkl','rb') as file1:
    train_data = pickle.load(file1)
print("Данные для обучения")
print(train_data)
with open('C:/Users/Danila/BERT/data/Train_pkl/training_answ.pkl','rb') as file2:
    train_labels = pickle.load(file2)
print("Метки для обучения")
print(train_labels)

# Загрузка валидационных данных
with open('C:/Users/Danila/BERT/data/Validation_pkl/val_list.pkl','rb') as file3:
    val_data = pickle.load(file3)
print("Данные для валидации")
print(val_data)
with open('C:/Users/Danila/BERT/data/Validation_pkl/val_answ.pkl','rb') as file4:
    val_labels = pickle.load(file4)
print("Метки для валидации")
print(val_labels)


# Инициализация токенизатора
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Преобразование тренировочных данных в формат, подходящий для модели
train_encodings = tokenizer(train_data, truncation=True, padding=True, max_length=128)
train_labels = torch.tensor(train_labels)

# Преобразование валидационных данных в формат, подходящий для модели
val_encodings = tokenizer(val_data, truncation=True, padding=True, max_length=128)
val_labels = torch.tensor(val_labels)

# Создание тренировочного и валидационного наборов данных
train_dataset = torch.utils.data.TensorDataset(train_encodings.input_ids, train_encodings.attention_mask, train_labels)
val_dataset = torch.utils.data.TensorDataset(val_encodings.input_ids, val_encodings.attention_mask, val_labels)

# Определение размера тренировочного и валидационного наборов данных
train_size = len(train_dataset)
val_size = len(val_dataset)

# Определение размера батча и количества эпох
batch_size = 30
num_epochs = 10

# Создание загрузчиков данных
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Инициализация модели
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Определение устройства для обучения (CPU или CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Инициализация оптимизатора
optimizer = AdamW(model.parameters(), lr=1e-5)
model.train()

criterion = nn.CrossEntropyLoss()
# Цикл обучения модели
best_accuracy = 0.0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    # Анализ прохождения валидационной базы данных по эпохам
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            logits = outputs.logits
            _, predicted = torch.max(logits, dim=1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total

    # Вывод информации о процессе обучения
    print(f'Epoch {epoch + 1}/{num_epochs}:')
    print(f'Train Loss: {train_loss / train_size:.4f}')
    print(f'Validation Loss: {val_loss / val_size:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}')

    # Сохранение модели с автоматической нумерацией названия
    model_name = f'model_{epoch + 1}.pt'
    torch.save(model.state_dict(), model_name)

    # Обновление лучшей точности
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model_name = model_name

# Вывод лучшей модели
print(f'Best Model: {best_model_name}')
