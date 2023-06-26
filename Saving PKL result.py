import pickle
import pandas as pd

data = pd.read_excel('Total_Val_FV_GPT.xlsx')
fruit_df = pd.DataFrame(data)
f_list = fruit_df.values.tolist()
fruit_list = []
fruit_answer_list = []
print(f_list)
print(type(f_list))
print(f_list[0][0])
#Узнаем объем всего датафрейми
def list_length(list):
   counter = 0
   for i in list:
       counter=counter+1
   return counter
counter = list_length(f_list)
print(counter)
for i in range(counter-1):
    fruit_list.append(f_list[i][0])
    fruit_answer_list.append(f_list[i][1])
print(fruit_list)
print(fruit_answer_list)
#Сохраняем результат в pickle
filename1 = 'val_list.pkl'
with open(filename1,'wb') as file1:
    pickle.dump(fruit_list, file1)

filename2 = 'val_answ.pkl'
with open(filename2,'wb') as file2:
    pickle.dump(fruit_answer_list, file2)

#
# # Чтение содержимого файла и загрузка в список
# with open(filename, 'rb') as file:
#     word_list = pickle.load(file)
#
# # Вывод списка слов
# print(word_list)