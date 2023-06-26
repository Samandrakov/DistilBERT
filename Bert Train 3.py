import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pickle

# Загрузка данных
class CustomDataset(Dataset):
    def __init__(self, data, labels, tokenizer):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }


# Объявление модели и токенизатора
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Загрузка данных обучающей и валидационной выборки

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

# Создание экземпляров Dataset
train_dataset = CustomDataset(train_data, train_labels, tokenizer)
val_dataset = CustomDataset(val_data, val_labels, tokenizer)

# Определение параметров обучения
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Определение оптимизатора и функции потерь
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Определение устройства для обучения (GPU или CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Перевод модели на устройство
model = model.to(device)

# Создание DataLoader для обучающей и валидационной выборки
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Функция для вычисления точности
def accuracy(preds, labels):
    _, predicted = torch.max(preds, dim=1)
    correct = (predicted == labels).float()
    acc = correct.sum() / len(correct)
    return acc


# Обучение модели
for epoch in range(num_epochs):
    print("Epoch",epoch)
    # Обучение
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)

        loss = criterion(outputs.logits, labels)
        acc = accuracy(outputs.logits, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += acc.item()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # Валидация
    model.eval()
    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)

            loss = criterion(outputs.logits, labels)
            acc = accuracy(outputs.logits, labels)

            val_loss += loss.item()
            val_acc += acc.item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

    # Вывод аналитики
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

# Сохранение модели
torch.save(model.state_dict(), 'model.pth')
