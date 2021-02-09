import requests, re, os
from pathlib import Path

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

subs = ['amitheasshole', 'unpopularOpinion', 'lifeprotips', 'relationships', 'confessions', 'tifu']
sName = {}
#for s in subs:
s = subs[0]
paths = sorted(Path(s).iterdir(), key=os.path.getmtime)
path = str(paths[-1])
path = path.replace(s+'\\', '')
path = path.replace('.txt', '')
sName[s] = path
print(path)

base = requests.get('https://api.pushshift.io/reddit/submission/search?ids='+path, headers=headers)
b = base.json()['data'][0]
stamp = b['created_utc']
while True:
    stamp += 3600
    for s in subs:
        if s not in os.listdir():
            os.mkdir(s)
        
        startEnd = s+"&after="+str(stamp)+"&before="+str(stamp+3600)+'&size=25.json'
        url='https://api.pushshift.io/reddit/submission/search?subreddit='+startEnd
        r = requests.get(url, headers=headers)
        h = {'data':[]}
        try:
            h = r.json()
        except:
            print("Connection timed out")
        posts = h['data']
        for val in posts:
            #print(val['selftext'])
            if 'selftext' in val.keys():
                raw = val['selftext']
            else:
                raw = '[removed]'
            if raw != '[removed]':
                raw = raw.replace('\\xe2\\x80\\x99', "'")
                raw = raw.replace('\\xe2\\x80\\x9c', '"')
                raw = raw.replace('\\xe2\\x80\\x9d', '"')
                raw = raw.replace('&amp;#39;', "'")
                raw = re.sub('&lt.+?&gt;', '', raw)
                toReplace = ['&lt;', '&gt;']
                for reg in toReplace:
                    raw = re.sub(reg, '', raw)
                name = val['id']
                
                if len(name) > 0:
                    name = re.sub('<id.*?_', '', name)
                    name = name.replace('</id>', '')
                    #print(name)
                    sName[s] = name
                    if not os.path.exists(s+"\\"+name+'.txt'):
                        print(name)
                        file=open(s+"\\"+name+'.txt', 'wb')
                        refs=file.write(raw.encode('utf8'))
                        file.close()
                    else:
                        print(name, 'exists already in sub', s)
                else:
                    print('No name found')
            else:
                print("Val is removed", val['id'], s)
            
        
        #time.sleep(2)