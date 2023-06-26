import pandas as pd
import json

data = pd.read_excel('Фрукты и овощи тестовый.xlsx')
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

# with open("fruit.json", 'w') as f:
#     # indent=2 is not needed but makes the file human-readable
#     # if the data is nested
#     json.dump(fruit_list, f, indent=2)
with open("FruVeg-answ.txt", "w") as output:
    output.write(str(fruit_answer_list))