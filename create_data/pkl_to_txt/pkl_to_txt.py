import pickle

pkl_file = 'widerface_train_annotation.pkl'

with open(pkl_file, 'rb') as f:  
    data = pickle.loads(f.read())

f = open('widerface_train_annotation.txt','w')

bbox = []

for title in data:
    for j in range(len(data[title])):
        bbox += data[title][j]['bbox']
        print(bbox)
    
    offset = 1
    temp = title
    while temp[temp.index('_')+1].isdigit()== False:
        print(temp[temp.index('_')+1])
        offset = offset + 1
        temp = temp[temp.index('_')+1:]
        
    title = title.replace('_','/',offset)
    if offset > 1:
        title = title.replace('/','_',offset-1)
    
    print(title)
    offset = 1
    
    f.write(str(title))
    
    f.write(' ')

    for x in bbox:
        f.write(x + ' ')
    
    f.write('\n')

    bbox = []


  
f.close()
    