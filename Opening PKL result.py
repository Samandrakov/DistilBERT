import pickle

with open('GPT_fvlist.pkl','rb') as file1:
    fvlist = pickle.load(file1)
print(fvlist)
with open('GPT_fvansw.pkl','rb') as file2:
    fvansw = pickle.load(file2)
print(fvansw)
