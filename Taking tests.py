import pandas as pd
import pickle

data = pd.read_excel('C:/Users/Danila/BERT/Names.xlsx')
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

filename1 = 'Names_Test1.plk'
with open(filename1,'wb') as file1:
    pickle.dump(fruit_list, file1)

filename2 = 'Answers_Test1.pkl'
with open(filename2,'wb') as file2:
    pickle.dump(fruit_answer_list, file2)

# with open("Names_Test1.txt", "w") as output:
#      output.write(str(fruit_list))
# with open("Answers_Test1.txt", "w") as output:
#     output.write(str(fruit_answer_list))