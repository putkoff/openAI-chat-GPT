import PySimpleGUI as sg
import functions as fun
import json
"""
    These 8 "Constructs" or Design Patterns demonstrate numerous ways of "generating" or building your layouts
    0 - A simple list comprehension to build a row of buttons
    1 - A simple list comprehension to build a column of buttons
    2 - Concatenation of rows within a layout
    3 - Concatenation of 2 complete layouts [[ ]] + [[ ]] = [[ ]]
    4 - Concatenation of elements to form a single row [ [] + [] + [] ] = [[ ]]
    5 - Questionnaire - Using a double list comprehension to build both rows and columns in a single line of code
    6 - Questionnaire - Unwinding the comprehensions into 2 for loops instead
    7 - Using the * operator to unpack generated items onto a single row 
    8 - Multiple Choice Test - a practical use showing list comprehension and concatenated layout
"""
def getDefs(js,jsDef):
    keys = fun.getKeys(js)
    for k in range(0,len(keys)):
        key = keys[k]
        jsDef[key] = js[key]
    return jsDef
def eatAll(x,ls):
    go = True
    
    while go == True:
        if len(x) == 0:
            go = False
        elif len(x) <=1 or x[0] not in ls:
            go = False
        elif x[0] in ls:
            x = x[1:]
    go = True
    while go == True:
        if len(x) == 0:
            go = False
        elif len(x) <=1 or x[-1] not in ls:
            go = False
        elif x[-1] in ls:
            x = x[:-1]
    return x
def addAll(x,ls):
    x,go = eatAll(x,ls),True
    while go == True:
        if len(x) == 0:
            go = False
        elif len(x) <1 or x[0] not in ls:
            go = False
        elif x[0] in ls:
            x = x[1:]
    go = True
    while go == True:
        if len(x) == 0:
            go = False
        elif len(x) <1 or x[-1] not in ls:
            go = False
        elif x[-1] in ls:
            x = x[:-1]
    x = ls[0]+str(x)+ls[0]
    return x
def mkJs(n,x,y):
    return n +addAll(x,['"',"'"])+'=js['+addAll(y,['"',"'"])+'],'
def mkJsReturn(n,x):
    return n +eatAll(x,['"',"'"])+'=js['+addAll(x,['"',"'"])+'],'
def mkReturnJS(js):
    n,keys = '',fun.getKeys(js)
    for i in range(0,len(keys)):
        n = mkJsReturn(n,eatAll(keys[i]))
    return n[:-1]
def mkReturnls(ls):
    name,sgName,ls = ls
    n,js = '',{}
    for i in range(0,len(ls)):
        print(ls[i])
        spl = fun.mkLs(ls[i].replace(' =','=').replace('= ','=').split('='))
        js[eatAll(spl[0],['"',"'"])] = ""
        if '=' in ls[i]:
            js[eatAll(spl[0],['"',"'"])] = boolIt(spl[1])
        n = mkJsReturn(n,spl[0])
    n = 'def '+name+'(js):\n\tjs = getDefs(js,'+str(js)+'):\n\treturn '+sgName+'('+n[:-1]+')'
    return n
def boolIt(x):
    if str(x) in ['None','True','False']:
        return bool(x)
    if [str(x)[0],str(x)[-1]] == ['(',')']:
        return eatAll(x,['"',"'"])
    return x
stri = '''Graph(canvas_size,
    graph_bottom_left,
    graph_top_right,
    background_color = None,
    pad = None,
    p = None,
    change_submits = False,
    drag_submits = False,
    enable_events = False,
    motion_events = False,
    key = None,
    k = None,
    tooltip = None,
    right_click_menu = None,
    expand_x = False,
    expand_y = False,
    visible = True,
    float_values = False,
    border_width = 0,
    metadata = None)'''
def cleanLs(ls):
    lsN = []
    for i in range(0,len(ls)):
        if ls[i] != '':
            lsN.append(ls[i])
    return lsN
def parseOut(x,ls):
    z,lsN,parse = '',[],True
    for i in range(0,len(x)):
        if x[i] in ls:
            if x[i] == ls[0]:
                parse = False
                cou = 0
            if x[i] == ls[1]:
                parse = True
    
        elif parse == False and cou == 0:
            lsN.append(eatAll(z.replace('\n    ',''),[ls[0],ls[1],'\n','\t']))
            lsN = cleanLs(fun.mkLs(lsN))
            z = ''
            cou = 1
        if parse == True:
            z = z + x[i]
    lsN.append(eatAll(x.split(ls[1])[-1],[ls[0],ls[1],'\n','\t']))
    return lsN
def ifNotInParse(x,ls,delim):
    name= x.split('(')[0]
    end = x.split(')')[-1]
    x = x[len(name+'('):-len(')'+end)]
    

    z,lsN,parse = '',[],True
    for i in range(0,len(x)):
        z = z + x[i]
        if x[i] == ls[0]:
            parse = False
        elif x[i] == ls[1]:
            parse = True
        elif x[i] == delim and parse == True:
            lsN.append(eatAll(z+x[i],['\n','\t',' ',',']))
            z = ''
    input(lsN)
    input('here')
    lsN.append(eatAll(x.split(ls[1])[-1],[ls[0],ls[1],'\n','\t']))
    input(lsN)
    return [name,'sg.'+name,lsN]
fun.pen('','newFuncs.py')
#input(mkReturnls(ifNotInParse(stri,['(',')'],',')))
spl = fun.reader('html.txt').split('<pre><code class="hljs python">')

for k in range(1,len(spl)):
    n,js = '',{}
    lsN = parseOut(spl[k].split('</code></pre>')[0],['<','>'])
    name= lsN[0].split('(')[0]
    end = lsN[-1].split(')')[-1]
    lsN[0] = lsN[0][len(name+'('):]
    for i in range(0,len(lsN)):
            if '=' in lsN[i]:
              eq = lsN[i].replace(' =','=').replace('= ','=').split('=')[0].split(' ')[-1]+'='
              lsN[i-1] = lsN[i-1]+lsN[i][:-len(eq)+1]
              if eq[0] == ',':
                eq = eq[1:]
              lsN[i] = eq
            print(lsN[i])

    name,sgName,lsN = ifNotInParse(n,['(',')'],',')
    for i in range(0,len(lsN)):
      if lsN[i] == ')':
        lsN[i-1] = lsN[i-1]+')'
        lsN[i] = ''
    lsN = cleanLs(fun.mkLs(lsN))
    for i in range(0,len(lsN)):
      lsN[i] = lsN[i].split(' = ')
      if fun.isLs(lsN[i]):
        js[lsN[i][0]] = boolIt(lsN[i][1].replace(" '("," (").replace(")', ","), "))
        n = n + str(lsN[i][0])+'=js['+str(lsN[i][0])+'],'
 
    n = 'def '+name+'(js):\n\tjs = getDefs(js,'+str(js)+'):\n\treturn '+sgName+'('+n[:-1]+')'
    fun.pen(fun.reader('newFuncs.py')+'\n'+str(n),'newFuncs.py')
