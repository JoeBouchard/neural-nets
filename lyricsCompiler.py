import os, re
count = 0
raw = open('lyrictrain/lyrictrain0.txt', 'a')
l = os.listdir('lyrics')
for name in l:
    if 'Refs.txt' not in name and '.txt' in name:
        file = open('lyrics/'+name, 'rb')
        r = file.read()
        r = re.sub('<.*?>', '', str(r))
        r = re.sub('[.*?]', '', r)
        r = re.sub('\n', '.', r)
        r = r.replace("b'", '')
        r = r.replace('\\r\\n', '.')
        if 'the' in r:
            if len(r) > 0:
                count += 1
                raw.write(".\n\n"+r)
        # else:
        #     print(name)
        file.close()
    
    if count%100 == 0:
        print(count)
    
    if count%10000 == 0:
        raw.close()
        raw = open('lyrictrain/lyrictrain'+str(int(count/10000))+".txt", 'a')

raw.close()
