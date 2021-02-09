import os
count = 0
raw = open('simplewikitrain/simplewikitrain0.txt', 'a')
l = os.listdir('simplewikiText')
for name in l:
    if 'Refs.txt' not in name and '.txt' in name:
        file = open('simplewikiText/'+name)
        r = file.read()
        toRemove = ['\\r\\n\\r\\n\\r\\nadditional terms may apply.  By using this site, you agree to the <a href="~`~`foundation.wikimedia.org~`wiki~`Terms_of_Use">Terms of Use<~`a> and <a href="~`~`foundation.wikimedia.org~`wiki~`Privacy_policy">Privacy Policy<~`a>. Wikipedia\\\\xc2\\\\xae is a registered trademark of the <a href="~`~`www.wikimediafoundation.org~`">Wikimedia Foundation, Inc.<~`a>, a non-profit organization.<~`li>',
            '\\r\\n\\r\\n\\r\\nPlease <a href="" title="Reload this page" onclick="window.location.reload(false); return false">try again<~`a> in a few&nbsp;minutes.<~`p>']
        for i in toRemove:
            r = r.replace(i, '')
        if len(r) > 0:
            count += 1
            raw.write("\n\n"+r)
        file.close()
    
    if count%100 == 0:
        print(count)
    
    if count%5000 == 0:
        raw.close()
        raw = open('simplewikitrain/simplewikitrain'+str(int(count/5000))+".txt", 'a')

raw.close()
