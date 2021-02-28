import os

def tokenize(sent):
    toReturn = []
    size=500
    for i in sent:
        toReturn.append(ord(i))

    while len(toReturn) < size:
        toReturn.append(0)
    return ', '.join(toReturn)+'\n'

    
l = os.listdir('trainingdata')
for i in l:
    f = open('trainingdata\\'+i, 'rb')#, encoding='utf-8')
    raw = f.read()
    f.close()
    split = raw.split(b'\\nXFLORB\\n')
    counter = 0
    app = open('trainingdata\\tokenized'+i, 'ab')
    for sent in split:
        app.write(tokenize(sent))
        if counter%1000 == 0:
            print(i, counter)
        counter+=1
        
