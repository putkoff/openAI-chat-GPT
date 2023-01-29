def ifNoThenApp(ls,x):
  if x not in ls:
    ls.append(x)
  return ls
def getAllNew(ls,ls2):
  for i in range(0,len(ls2)):
    ls = ifNoThenApp(ls,ls2[i])
  return ls
def getKeys(js):
  lsN = []
  try:
    for key in js.keys():
      lsN.append(key)
    return lsN
  except:
    return lsN
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
        
        opt = ['required','optional']
        inIt = False
        for c in range(0,len(opt)):
          if lsN[l] in paramNeeds[keys[i]][keys2[k]][opt[c]]:
            vars2[keys[i]][keys2[k]][str(lsN[l])] = str(parameters[lsN[l]]['default'])
            input([vars2[keys[i]][keys2[k]][str(lsN[l])],keys2[k]])
            inIt = True
            #input([vars2,paramNeeds[keys[i]][keys2[k]][opt[c]]])
        #if inIt == False:
          #paramNeeds[keys[i]][keys2[k]][opt[c]].append(lsN[l])
  print(vars2)
  return vars2
global alls,paramNeeds,parameters

parameters = {'model': {'object': 'str', 'default': 'text-davinci-003', 'scale': 'array', 'array': ['completion', 'edit', 'code', 'embedding']}, 'max_tokens': {'object': 'int', 'scale': 'range', 'range': {0: 2048}, 'default': 2000}, 'logit_bias': {'object': 'map', 'scale': 'range', 'range': {-100: 100}, 'default': None}, 'size': {'object': 'str', 'default': '1024x1024', 'scale': 'choice', 'choices': ['256x256', '512x512', '1024x1024']}, 'temperature': {'object': 'float', 'default': 0.7, 'scale': 'range', 'range': {-2.0: 2.0}}, 'best_of': {'object': 'int', 'default': 1, 'scale': 'range', 'range': {0: 10}}, 'top_p': {'object': 'float', 'default': 0.0, 'scale': 'range', 'range': {0.0: 1.0}}, 'frequency_penalty': {'object': 'float', 'default': 0.0, 'scale': 'range', 'range': {-2.0: 2.0}}, 'presence_penalty': {'object': 'float', 'default': 0.0, 'scale': 'range', 'range': {-2.0: 2.0}}, 'log_probs': {'object': 'int', 'default': 1, 'scale': 'range', 'range': {1: 10}}, 'stop': {'object': 'str', 'default': None, 'scale': 'array', 'range': {0: 4}}, 'echo': {'object': 'bool', 'default': False, 'scale': 'choice', 'choice': [True, False]}, 'n': {'object': 'int', 'default': 1, 'scale': 'range', 'range': {1: 10}}, 'stream': {'object': 'bool', 'default': False, 'scale': 'choice', 'choice': [True, False]}, 'suffix': {'object': 'str', 'default': None, 'scale': 'range', 'range': {0: 1}}, 'prompt': {'object': 'str', 'default': None, 'scale': 'inherit'}, 'input': {'object': 'str', 'default': None, 'scale': 'inherit'}, 'instruction': {'object': 'str', 'default': None, 'scale': 'inherit'}, 'response_format': {'object': 'str', 'default': 'url', 'scale': 'choice', 'choice': ['url', 'b64_json']}, 'image': {'object': 'str', 'default': None, 'scale': 'upload', 'upload': {'type': ['PNG', 'png'], 'size': {'scale': {0: 4}, 'allocation': 'MB'}}}, 'mask': {'object': 'str', 'default': None, 'scale': 'upload', 'upload': {'type': ['PNG', 'png'], 'size': {'scale': {0: 4}, 'allocation': 'MB'}}}, 'file': {'object': 'str', 'default': None, 'scale': 'upload', 'upload': {'type': ['jsonl'], 'size': {'scale': {0: 'inf'}}, 'allocation': 'MB'}}, 'purpose': {'object': 'str', 'default': None, 'scale': 'inherit'}, 'file_id': {'object': 'str', 'default': None, 'scale': 'inherit'}, 'user': {'object': 'str', 'default': 'defaultUser', 'scale': 'inherit'}}
vars = {'completion': {'create': ['model', 'prompt', 'user', 'stream', 'n', 'suffix', 'max_tokens', 'temperature', 'best_of', 'top_p', 'frequency_penalty', 'presence_penalty', 'log_probs', 'stop', 'echo']},
        'embedding': {'create': ['model', 'input', 'user']}, 'image': {'create': ['prompt', 'user', 'size', 'n', 'response_format'],
                                                                       'edit': ['image', 'mask', 'user', 'prompt', 'size', 'n', 'response_format'],
                                                                       'variation': ['image', 'prompt', 'mask', 'user', 'size', 'n', 'response_format']},
        'edit': {'create': ['model', 'input', 'user', 'instruction']},
        'moderation': {'moderate': ['model', 'input', 'user']}}
paramNeeds = {'completion':{'create':{'required':{'model','prompt'},'optional':{'user','stream','n','suffix','max_tokens','logit_bias','temperature','best_of','top_p','frequency_penalty','presence_penalty','log_probs','stop','echo'}}},
         'edit':{'create':{'required':{'model','instruction'},'optional':{'user','input','n','temperature','top_p'}}},
         'image':{'create':{'required':{'prompt'},'optional':{'user','size','n','response_format'}},
                  'edit':{'required':{'image','prompt'},'optional':{'mask','user','size','n','response_format'}},
                  'variation':{'required':{'image'},'optional':{'mask','user','size','n','response_format'}}},
         'embedding':{'create':{'required':{'model','input'},'optional':{'user'}}},
         'moderation':{'moderate':{'required':{'input'},'optional':{'model','user'}}}}
        
getallDefs()
{'completion': {'create': {'model': 'text-davinci-003', 'prompt': 'None', 'user': 'defaultUser', 'stream': 'False', 'n': '1', 'suffix': 'None', 'max_tokens': '2000', 'temperature': '0.7', 'best_of': '1', 'top_p': '0.0', 'frequency_penalty': '0.0', 'presence_penalty': '0.0', 'log_probs': '1', 'stop': 'None', 'echo': 'False'}},
 'embedding': {'create': {'model': 'text-davinci-003', 'prompt': 'None', 'user': 'defaultUser', 'stream': 'False', 'n': '1', 'suffix': 'None', 'max_tokens': '2000', 'temperature': '0.7', 'best_of': '1', 'top_p': '0.0', 'frequency_penalty': '0.0', 'presence_penalty': '0.0', 'log_probs': '1', 'stop': 'None', 'echo': 'False', 'input': 'None'}}, 'image': {'create': {'model': 'text-davinci-003', 'prompt': 'None', 'user': 'defaultUser', 'stream': 'False', 'n': '1', 'suffix': 'None', 'max_tokens': '2000', 'temperature': '0.7', 'best_of': '1', 'top_p': '0.0', 'frequency_penalty': '0.0', 'presence_penalty': '0.0', 'log_probs': '1', 'stop': 'None', 'echo': 'False', 'input': 'None', 'size': '1024x1024', 'response_format': 'url'}, 'edit': {'model': 'text-davinci-003', 'prompt': 'None', 'user': 'defaultUser', 'stream': 'False', 'n': '1', 'suffix': 'None', 'max_tokens': '2000', 'temperature': '0.7', 'best_of': '1', 'top_p': '0.0', 'frequency_penalty': '0.0', 'presence_penalty': '0.0', 'log_probs': '1', 'stop': 'None', 'echo': 'False', 'input': 'None', 'size': '1024x1024', 'response_format': 'url', 'image': 'None', 'mask': 'None'}, 'variation': {'model': 'text-davinci-003', 'prompt': 'None', 'user': 'defaultUser', 'stream': 'False', 'n': '1', 'suffix': 'None', 'max_tokens': '2000', 'temperature': '0.7', 'best_of': '1', 'top_p': '0.0', 'frequency_penalty': '0.0', 'presence_penalty': '0.0', 'log_probs': '1', 'stop': 'None', 'echo': 'False', 'input': 'None', 'size': '1024x1024', 'response_format': 'url', 'image': 'None', 'mask': 'None'}}, 'edit': {'create': {'model': 'text-davinci-003', 'prompt': 'None', 'user': 'defaultUser', 'stream': 'False', 'n': '1', 'suffix': 'None', 'max_tokens': '2000', 'temperature': '0.7', 'best_of': '1', 'top_p': '0.0', 'frequency_penalty': '0.0', 'presence_penalty': '0.0', 'log_probs': '1', 'stop': 'None', 'echo': 'False', 'input': 'None', 'size': '1024x1024', 'response_format': 'url', 'image': 'None', 'mask': 'None', 'instruction': 'None'}}, 'moderation': {'moderate': {'model': 'text-davinci-003', 'prompt': 'None', 'user': 'defaultUser', 'stream': 'False', 'n': '1', 'suffix': 'None', 'max_tokens': '2000', 'temperature': '0.7', 'best_of': '1', 'top_p': '0.0', 'frequency_penalty': '0.0', 'presence_penalty': '0.0', 'log_probs': '1', 'stop': 'None', 'echo': 'False', 'input': 'None', 'size': '1024x1024', 'response_format': 'url', 'image': 'None', 'mask': 'None', 'instruction': 'None'}}}
