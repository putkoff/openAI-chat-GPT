import os
import openai
import requests
import json
import gui
import cv2
openai.api_key = <insert your apy key>
def changeGlob(x,v):
    globals()[x] = v
def getFile():
  return gui.fileBrowser()
def mkDir(x):
  if isDir(x) == False:
    os.mkdir(x)
  return x
def crPa(x,y):
  return os.path.join(x,y)
def isDir(x):
  return os.mkdir(x)
def isFile(x):
  return os.path.isfile(x)
def pen(x,p):
    with open(p, 'w',encoding='UTF-8') as f:
        return f.write(str(x))
def reader(x):
    with open(x, 'r',encoding='UTF-8') as f:
        return f.read()
def jsRead(x):
  return json.loads(reader(x))
def jsItRead(x):
  return jsIt(reader(x))
def jsIt(js):
  return json.loads(str(js).replace("'",'"'))
def jsItSpec(js):
  z,x,inner = '',str(js),False
  for i in range(0,len(x)):
    y = x[i]
    if x[i] == '<':
      inner = True
    if x[i] == '>':
      inner = False
    if inner == False and x[i] == "'":
      y = '"'
    z = z + y
  return json.loads(z)
      
def makeQuote(x):
  for i in range(0,2):
    x = whileIn(x,0-i,['"',"'"])
  return '"'+x+'"'
def whileIn(x,n,ls):
  if n == -1:
    while x[n] in ls:
      x = x[:-1]
  elif n == 0:
    while x[n] in ls:
      x = x[1:]
  return x
def numLs():
  return str('0,1,2,3,4,5,6,7,8,9,0').split(',')
def isNum(x):
  if isInt(x):
    return True
  if isFloat(x):
    return True
  if isStr(x) == False:
    x = str(x)
  num,cou = numLs(),0
  for i in range(0,len(x)):
    if x[i] not in num:
      if x[i] != '.':
        return False
      elif x[i] == '.' and cou >0:
        return False
      elif x[i] == '.' and cou ==0:
        cou +=1
  return True
def isLs(ls):
    if type(ls) is list:
        return True
def isStr(x):
  if type(x) is str:
    return True
def isInt(x):
  if type(x) is int:
    return True
def isFloat(x):
  if type(x) is float:
    return True
def isBool(x):
    if type(x) is bool:
        return True
def mkFloat(x):
  if isFloat(x):
    return x
  if isInt(x):
    return float(str(x))
  if isNum(x):
    return float(str(x))
  z = ''
  for i in range(0,len(x)):
    if isNum(x[i]):
      z = z + str(x[i])
  if len(z) >0:
    return float(str(z))
  return float(str(1))
def mkBool(x):
    if isBool(x):
        return x
    boolJS = {'0':'True','1':'False','true':'True','false':'False'}
    if str(x) in boolJS:
        return bool(str(boolJs[str(x)]))
    return None
def mkStr(x):
    if isStr(x):
        return x
    return str(x)
def getObjObj(obj,x):
    if obj in ['str','file','image','mask','input','prompt']:
        return mkStr(x)
    if obj == 'bool':
        return mkBool(x)
    if obj == 'float':
        return mkFloat(x)
    if obj == 'int':
        return int(x)
    if obj == 'map':
        return x
    return x
def getKeys(js):
  lsN = []
  try:
    for key in js.keys():
      lsN.append(key)
    return lsN
  except:
    return lsN
def isLs(ls):
  if type(ls) is not list:
    ls = [ls]
  return ls
def getallDefs():
  jsN,vars2,lsN,keys = {},{},[],getKeys(vars)
  for i in range(0,len(keys)):
    var = vars[keys[i]]
    vars2[keys[i]] = {}
    keys2 = getKeys(var)
    for k in range(0,len(keys2)):
      lsN = getAllNew(lsN,vars[keys[i]][keys2[k]])
      vars2[keys[i]][keys2[k]] = {}
      for l in range(0,len(lsN)):
        vars2[keys[i]][keys2[k]][str(lsN[l])] = str(parameters[lsN[l]]['default'])
        opt = ['required','optional']
        inIt = False
        for c in range(0,len(opt)):
          if lsN[l] in paramNeeds[keys[i]][keys2[k]][opt[c]]:
            inIt = True
        if inIt == False:
          paramNeeds[keys[i]][keys2[k]][opt[c]].append(lsN[l])
  return vars2
def getJson(x):
  return x.json()
def getText(x):
  return x.text
def callJson(ls):
  return getJson(getCall(ls))
def callText(ls):
  return getText(getCall(ls))
def uploadFile(file,purpose):
  return openai.File.create(file=open(file, "rb"),purpose=purpose)
def retrieveFile(id):
  return openai.File.retrieve(id)
def retrieveContent(id):
  return openai.File.download(id)
def listFiles():
  return openai.File.list()
def GETHeader():
  return {"Content-Type": "application/json" ,"Authorization": "Bearer "+openai.api_key,}
def reqGet(ls):
  return requests.get(ls[0],json=ls[1],headers=GETHeader())
def reqPost(ls):
  return requests.post(ls[0], json=ls[1], headers=GETHeader())
def reqDelete(ls):
  return requests.delete(ls[0], json=ls[1], headers=GETHeader())
def reqImageVariation(js):
  return openai.Image.create_variation(image=open(str(js['image']), "rb"),n=js['n'],size=js['size'],prompt = js['prompt'],response_format=js['response_format'])
def reqImageEdit(js):
  return openai.Image.create_edit(image=open(str(js['image']), "rb"),mask=open(str(js['mask']), "rb"),prompt = js['prompt'],n=js['n'],size=js['size'],response_format=js['response_format'])
def retrieveModel(model):
  return reqGet([model,None])
def getModels():
  pen(callJson(None,endpoints['models']['list']),'modelsList.json')
def getEngines():
  pen(callJson(None,endpoints['engines']['list']),'enginesList.json')
def getFiles():
  pen(callJson(None,endpoints['files']['list']),'filesList.json')
def getSends(js,ls):
  return js[ls[0]][ls[1]]
def correctJson(x):
  pen(reader(x).replace('"default":,','"default":"None",').replace("'",'"').replace('bool(""True"")','True').replace('bool(""False"")','False').replace('"None"','None').replace('None','"None"').replace('"False"','False').replace('"True"','True').replace('""','').replace('False','"False"').replace('True','"True"'),x)
  return jsRead(x)
def getAllDefaults(ls):
  for i in range(0,len(ls)):
    '"'+str(ls[i])+'"='+parameters[ls[i]]
def boolAsk(ls,st):
  ans = None
  while ans not in ['0','1','2']:
    ans = input(st+'\n0) default\n1) '+str(ls[0])+'\n2) '+str(ls[1]))
  if str(ans) == '0':
    return   
  return ans
def rmFromJs(js,x):
  jsN,keys = {},getKeys(js)
  for i in range(0,len(keys)):
    key = keys[i]
    if key != x:
      jsN[key] =getObjObj(parameters[key]['object'],js[key])
  return jsN
def ifEqrmFromJs(js,x):
  jsN,keys = {},getKeys(js)
  for i in range(0,len(keys)):
    key = keys[i]
    if str(x) not in str(js[key]):
      jsN[key] =getObjObj(parameters[key]['object'],js[key])
  return jsN
def getContext(scal,key,opt,default,type,object):
    ran = 'range'
    if scal == 'range':
      ran = 'between : '
    if scal == 'choice':
      ran = 'choices:'
    return '^^^^^^^^'+str(key)+'^^^^^^^^\n'+str(opt)+'\ndefault = '+str(default)+'\ntype: '+str(type)+'\nobject: '+str(object)+f'\nscale: {ran}:\nplease input your response for '+str(key)+'(leave blank for default):\n'
def getPrompt(scal,js):
  keys = getKeys(js['prompt'])
  prom =''
  for i in range(0,len(keys)):
    key = keys[i]
    part =''
    text,cycle = js['prompt'][key]['text'],js['prompt'][key]['cycle']
    if cycle == 'while':
      end = False
      while end == False:
        part = part + input(getContext(scal,key,'required','','str','string'))+','
        if boolAsk(['yes','no'],'would you like to add another?') == 1:
          end = True
    else:
      for i in range(0,int(cycle)):
        part = part + input(getContext(scal,key,'required','','str','string'))+','
    prom = prom + text + part
  return prom
def getPromptVars(ls):
  specs = ls[-1]
  promp = getSends(prompts,ls)
  keys = getKeys(promp)
  needs,req = getSends(paramNeeds,ls),['required','optional']
  boo  = boolAsk(['yes','no'],'include only required?')
  for k in range(0,2):
    opt = req[k]
    for i in range(0,len(promp)):
      key,ans = keys[i],''
      scal,obj = parameters[key]['scale'],parameters[key]['object']
      st = getContext(scal,key,opt,promp[key],scal,obj)
      if key in needs[opt]:
        if k!=int(boo):
          if key in ['input','prompt']:
            ans = getPrompt(scal,specs)
          elif key in ['file']:
            changeGlob('contType',"multipart/form-data")
            ans = reader(gui.fileBrowser())
          elif key in ['image','mask']:
            changeGlob('contType',"multipart/form-data")
            ans = gui.fileBrowser()
          elif obj == 'bool':
            ans = boolAsk(['True','False'],st)
          elif key == 'model':
            ans = askIt(specs['models']['choices'],st)
          elif scal == 'choices':
            input([parameters[key],key])
            ans = askIt(parameters[key][scal],st)
          else:
            ans = input(st)
      if str(ans) == '':
        ans = promp[key]
      promp[key] = getObjObj(parameters[key]['object'],ans)   
  promp = ifEqrmFromJs(promp,None)
  return promp
def getCall(ls):
  endsp,data = getSends(endpoints,ls),getPromptVars(ls)
  type = endsp['type']
  if 'var' in ends:
    endsp = endsp.replace(endsp['var',ls[1]])
    data = None
  if ls[0] =='image' and ls[1] == 'variation':
    return reqImageVariation(jsItSpec(data))
  if ls[0] =='image' and ls[1] == 'edit':
    return reqImageEdit(jsItSpec(data))
  ls = [endsp['endpoint'],jsItSpec(data)]
  if type == 'DELETE':
    response = reqDelete(ls)
  elif type == 'POST':
    response = reqPost(ls)
  elif type == 'GET':
    response = reqGet(ls)
  return response
def ifNoThenApp(ls,x):
  if x not in ls:
    ls.append(x)
  return ls
def getAllNew(ls,ls2):
  for i in range(0,len(ls2)):
    ls = ifNoThenApp(ls,ls2[i])
  return ls
def createAsk(ls):
  n = '0) default\n'
  for i in range(0,len(ls)):
    n = n + str(i+1)+') '+str(ls[i])+'\n'
  return n
def findIt(ls,x):
  for i in range(0,len(ls)):
    if x == ls[i]:
      return i
  return None
def askIt(ls,st):
  ans = -1
  while int(ans) not in range(0,len(ls)+1):
    ans = input(st+'\n'+createAsk(ls))
    if isInt(ans) == False:
      ans = -1
  if ans == 0:
    return ''
  return ls[int(ans)-1]
def askBoth():
  ls = [askIt(ends,'what are you looking to do?\n')]
  ls.append(askIt(getKeys(endpoints[ls[0]]),'which endpoint would you like to use?\n'))
  ls.append(specifications[ls[0]][askIt(getKeys(specifications[ls[0]]),'what specification are you looking for?\n')])
  return ls
from PIL import Image
global endpoints,models,engines,files,ends,parameters,vars,paramNeeds,prompts,specifications,contType
contType="application/json"
from pars import endpoints,models,engines,files,parameters,vars,paramNeeds,prompts,ends,specifications
changeGlob('contType',"multipart/form-data")
print(callJson(askBoth()))
