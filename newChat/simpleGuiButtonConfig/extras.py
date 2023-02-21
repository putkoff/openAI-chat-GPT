def pieceWhileNotIn(ls,delim):
    lsN,c,n = ls,1,''
    for k in range(0,len(ls)):
        if delim not in ls[k]:
            n = n + ls[k]
            lsN = lsN[1:]
            c+=1
        else:
            return n,k
    return n,k
#input(mkReturnls(ifNotInParse(stri,['(',')'],',')))
spl = fun.reader('html.txt').split('<pre><code class="hljs python">')
fun.pen('','newFuncs.py')
for k in range(1,len(spl)):
    n,js = '',{}
    
    lsN = parseOut(spl[k].split('</code></pre>')[0],['<','>'])
    name = lsN[0].split('(')[0]
    sgName = 'sg.'+name
    lsN[0] = lsN[0][len(name)+1:]
    input(lsN)
    lsA = lsN
    for i in range(0,len(lsN)):
       if ' =' in lsN[i]:
           arg = lsN[i].split(' =')[0]
           lsN[i] =lsN[i].split(' =')[1]
           new,c = pieceWhileNotIn(lsA[i:],' =')
           if len(lsN)> i+c+1:
               if lsN[i+c+1][0] == ')':
                   new = new + ')'
                   
                   lsN[i+c+1] = eatAll(lsN[i+c+1][1:],['\n','\t',' '])
           n = n + lsN[i].split(' =')[0]+'=js["'+lsN[i].split(' =')[0]+'"],'
           i +=c
       else:
            js[lsN[i]] = '""'
            n = n + lsN[i]+'=js["'+lsN[i]+'"],'
        
    n = 'def '+name+'(js):\n\tjs = getDefs(js,'+str(js)+'):\n\treturn '+sgName+'('+n[:-1]+')'
    fun.pen(fun.reader('newFuncs.py')+'\n'+str(n),'newFuncs.py')
