import requests, re, os, threading

def getArticle(name):
    name = name.replace(" ", "_")
    toRemove = ['\\','"', '?', ':', ',', '.', '*', '!', '>', '-']
    for c in toRemove:
        name = name.replace(c, '')
    print(name)
    if os.path.exists("simplewikiText\\"+name+'.txt'):
        try:
            file=open("simplewikiText\\"+name+'Refs.txt')
            refs=file.read()
            file.close()
        except:
            print('No refs found for '+name)
            return ['United States']
    else:
        r = requests.get('https://simple.wikipedia.org/w/index.php?title='+name+'&action=raw')
        raw = str(r.content)
        
        paragraphs = raw.split('\\n')
        
        toWrite = ''
        refs = ''
        for p in paragraphs:
            if p and p[0].isalnum() and len(p) > 100:
                #print(p)
                p=p.replace('/', '~`')
                p = re.sub(r"<ref.*<~`ref>", '', p)
                p = re.sub(r"<ref.*>", '', p)
                p = re.sub(r'\[\[[^\]\]]+?\|', '', p)
                newRefs = re.findall('\[\[.+?\]\]', p)
                for n in newRefs:
                    #print(n)
                    refs+=n+'\n'
                p = p.replace('[[', '')
                p = p.replace(']]', '')
                p = re.sub('{{.*}}', '', p)            
                toWrite+="\n"+p
        #print(toWrite)
        file = open("simplewikiText\\"+name+".txt", 'w')
        file.write(toWrite)
        file.close()
        file=open("simplewikiText\\"+name+"Refs.txt", 'w')
        file.write(refs)
        file.close()
    return(refs)

refs = ['Linguistics']
going = []
textRefs = []
while True:
    for r in refs:
        if len(going) < 5:
            going.append(threading.Thread(target=textRefs.append,args=(getArticle(r),)))
            going[-1].start()
            #print(len(going))
        else:
            going[0].join()
            active=0
            for i in going:
                if i.is_alive():
                    active+=1
            print(active)
            going.pop(0)
            newRefs = textRefs[0].replace('[[', '').replace(']]', '').split("\n")
            for n in newRefs:
                if n not in refs:
                    refs.append(n)
            textRefs.pop(0)
            