import os, re
subs = ['explainlikeimfive', 'askscience', 'askhistorians']#['unpopularOpinion', 'amitheasshole', 'tifu', 'lifeprotips', 'relationships', 'confessions']
for i in subs:
    files = os.listdir(i)
    l = len(files)
    save = open(i+str(l)+'.txt', 'a')
    print(i)
    counter = 0
    for j in files:
        file = open(i+'/'+j, 'rb')
        raw = re.sub(']\(.*?\)', '', str(file.read()))
        if raw != "b'deleted]'" and "Thank you for your submission!" not in raw:
            counter+=1
            save.write('\n\n'+str(raw))
            file.close()
            if counter%300 == 0:
                print(counter)
    save.close()