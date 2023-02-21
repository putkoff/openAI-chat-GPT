import functions as fun
import os
import requests
import json
import openai
openai.api_key = "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"
import PySimpleGUI as sg
import sys
from base64 import b64encode
import base64
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from datetime import date
def getDate():
    return date.today()
def changeGlob(x,y):
    globals()[x]=y
    return y
def urlToImg(x):
    from PIL import Image
    import requests
    url = x
    response = requests.get(url,stream=True)
    img = Image.open(response.raw)
    img.show()
    return img
def generate_from_masked_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read())
def resizeImg(x,y,z):
    image = openImg(x)
    size = image.size
    if size !=(y,z):
        image = Image.open(x)
        image = image.resize((y, z))
    return image
def openImg(x):
    return Image.open(x)
def streamImage(x):
    byte_stream = BytesIO()
    x.save(byte_stream, format='PNG')
    x = byte_stream.getvalue()
    return x
def imgScreen(x):
    return [sg.Image(data=x, key="-ArtistAvatarIMG-")]
def changeGlob(x,y):
    globals()[x]=y
    return y
def getKeys(js):
    return fun.getKeys(js)
def getVals(js):
    return fun.getVals(js)
def jsGet(js,x):
    if x in js:
        js = js[x]
    return js
def getNewKeys(js,x):
    js = jsGet(js,x)
    return getKeys(js),js
def regetJs(js):
    return jsOg
def ifOnlyOne(js,var):
    if fun.isLs(js[var]):   
        if len(js[var])> 1:
            return js[var]
    return js[js[var]]
def mkJs(js,key,sel):
    jsN = {}
    if sel !=None:
        js = js[sel]
    jsN[key] = getKeys(js)
    return jsN
def ifInJsWLs(ls,js,ls2):
    for i in range(0,len(ls)):
        if ls[i] in js:
            ls2[int(i)] = js[ls[int(i)]]
    return ls2
def ifInJs(js,var):
    return js[var]
def modelSpec(na):
        default,ls = ifInJsWLs(['default','list'],getChoices()[na],['',[]])
        specializedSet['jsList'].append(mkFull(na,mkType([default,specializedSet['paramJs']['object']]),ls,specializedSet['opt'],True,'drop'))
def ifLang(i,na):
    if i == 0:
        return na
    
    elif typ == 'input':
        ls = txtInput(na,ls[0],ls[1])
    elif typ == 'file':
        guiFuns.getBrwsSect(na,parameters[na]['description'],os.getcwd())
        ls = getFileBrowse(na,ls[0],ls[1],ls[2])
def promptSpec(promptJS):
    global category,specializations,categories,specialization,catKeys,specializedSet,catLs,categoriesJs
    if 'vars' in promptJS:
        vars = fun.getKeys(promptJS['vars'])
    vars,ls = fun.mkLs(vars),[]
    for i in range(0,len(vars)):
        if promptJS['vars'][vars[i]]['type'] == 'choice':
            modelSpec(vars[i])
        #elif promptJS['vars'][vars[i]]['type'] == 'list':
        #    specializedSet['jsList'].append([[sg.Text(promptJS['vars'][vars[i]]['input'])],[[sg.Input(ifLang(row,vars[i]),size=(15,5), pad=(4,4),key = vars[i]) for col in range(1)] for row in range(4)]])
        elif promptJS['vars'][vars[i]]['type'] in ['text','str','list']:
            
            specializedSet['inputTabKeys']['descriptions'].append(promptJS['vars'][vars[i]]['input'])
            specializedSet['inputTabKeys']['names'].append(vars[i])
            specializedSet['inputTabKeys']['types'].append(promptJS['vars'][vars[i]]['type'])
            specializedSet['inputTabKeys']['index']+=1
            #specializedSet['jsList'].append(mkFull(vars[i],'',[10,5],specializedSet['opt'],True,'input'))
def getlsFromBrac(x,obj):
    ls = str(x).replace('{','').replace('}','').replace(' ','').replace(':',',').split(',')
    for i in range(0,len(ls)):
        ls[i] = mkType([ls[i],obj])
    return ls
def getParamMenu(na,js):
    specializedSet['paramJs'] = js
    if na in getChoices():
        modelSpec(na)
    elif na in ['prompt','input']:
        promptSpec(returnParameters(category,specialization)['prompt'])
    else:
        defa,obj,scl = js['default'],js['object'],js['scale']
        if scl == 'upload':
            specializedSet['content'] = 'multipart/form-data' 
            specializedSet['jsList'].append(mkFull(na,'',[js['baseType'],'png.',os.getcwd(),parameters[na]['description']],specializedSet['opt'],True,'file'))
        elif js['object'] == 'bool':
            specializedSet['jsList'].append(mkFull(na,defa,na,specializedSet['opt'],True,'check'))
            
        elif js['object'] in ['float','int']:
          specializedSet['jsList'].append(mkFull(na,mkType([defa,obj]),getlsFromBrac(str(js['range']),js['object']),specializedSet['opt'],True,'slide'))
def selKeys():
    return getKeys(selections)
def ifLsSel(js,key):
    if selections[key] == '':
        return getKeys(js)
    return js[selections[key]]
def ifLsInKeyConc(keys,keyNs,js):
    for k in range(0,len(keys)):
        if returnParameters(category,specialization)['prompt']['vars'][keys[k]]['type'] == 'list':
            n = keys[k]
            for i in range(0,len(keyNs)):
                if str(keys[k])+str(i) in keyNs:
                    if js[str(keys[k])+str(i)] != None and js[str(keys[k])+str(i)] != "None":
                        n = n + ','+str(js[str(keys[k])+str(i)])
            js[keys[k]] = n
    return js
def tryJsTxt(resp):
    for i in range(0,2):
        try:
            js = resp.json()
            return js
        except:
            try:
                txt = resp.text
                return txt
            except:
                print(resp)
    return resp
def whichIsIn(js,ls):
    for i in range(0,len(ls)):
        if ls[i] in getKeys(js):
            return ls[i]
    return ls[0]
def compilePrompt(js):
    ifLsInKeyConc(fun.getKeys(returnParameters(category,specialization)['prompt']['vars']),fun.getKeys(js),js)
    vars,jsN = fun.getKeys(returnParameters(category,specialization)['prompt']['vars']),{}
    if len(vars)>0:
        isIn = whichIsIn(js,['prompt','input'])
        jsN[isIn] = returnParameters(category,specialization)['prompt']['task']+'\n'
        for i in range(0,len(vars)):
            if fun.isLs(js[vars[i]]):
                input(js)
                js[vars[i]] = str(js[vars[i]])[1:-1]
            jsN[isIn] = str(jsN[isIn])+str(returnParameters(category,specialization)['prompt']['vars'][vars[i]]['delimiter'])+str(js[vars[i]])+'\n'     
    prevKeys = specializedSet['prevKeys']
    for i in range(0,len(prevKeys)):
        if prevKeys[i] not in js:
            js[prevKeys[i]] = getAllInfo('parameters')[prevKeys[i]]['default']
        if js[prevKeys[i]] != None and js[prevKeys[i]] != 'None' and js[prevKeys[i]] != '':
            jsN[prevKeys[i]] = ifNotIntFl(js,prevKeys[i])
    try:
        return json.dumps(jsN)
    except:
        return jsN

def getParamDrop(ls):
    category,specialization=ls
    parametsCurr = returnParameters(category,specialization)
    keys= getKeys(parametsCurr)
    opt = ['required','optional']
    
    for i in range(0,len(opt)):
        specializedSet['opt'] = [True,False][i]
        paramLs = parametsCurr[opt[i]]
        for k in range(0,len(paramLs)):
            param = paramLs[k]
            specializedSet['prevKeys'].append(param)
            getParamMenu(param,getAllInfo('parameters')[param])
    specializedSet['inputTabKeys']['inputLs'] = [[sg.Tab(str(specializedSet['inputTabKeys']['names'][i]) if i < len(specializedSet['inputTabKeys']['names']) -1 else '+', tab(i), key=str(specializedSet['inputTabKeys']['names'][i])) for i in range(len(specializedSet['inputTabKeys']['names']))]]
    return  specializedSet['jsList']
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
  return {"Content-Type": specializedSet['content'] ,"Authorization": "Bearer "+openai.api_key}
def reqGet(js):
  return requests.get('https://api.openai.com/v1/completions',json= json.loads(js),headers=GETHeader())
def reqPost(js):
    js = json.loads(js)
    if specializedSet['content'] == 'multipart/form-data':
        if 'size' in js:
            sizeN = int(str(js['size']).split('x')[0])                    
        if selections['specialization'] == 'image_edit':
          return openai.Image.create_edit(image=open(resizeImg(str(js['image']),sizeN,sizeN),'rb'),mask=open(resizeImg(str(js['mask']),sizeN,sizeN),'rb'),prompt = js['prompt'],n=js['n'],size=js['size'],response_format=js['response_format'])
        if selections['specialization'] == 'image_variation':
            return openai.Image.create_variation(image=open(resizeImg(str(js['image']),sizeN,sizeN),'rb'),n=js['n'],size=js['size'],response_format=js['response_format'])
    resp = json.dumps(requests.post(getEndPoint(),json=js,headers=GETHeader()).json())
    fun.pen(resp,'ans.json')
    getResponse()
def tryJs(js):
    ls = [js,str(js),str(js).replace('"',"'"),str(js).replace('"','*&*').replace("'",'"')]
    for i in range(0,len(ls)):
        try:
            z = json.loads(ls)
            return z
        except:
            print('strike')
    return False        
def isDict(js):
    if fun.isStr(js) or fun.isInt(js) or fun.isFloat(js) or fun.isLs(js) or fun.isBool(js) or len(getKeys(js))==0 or tryJs(js) == False:
        return False
    return True
def ifThenMkJs(js):
    if isDict(js) != False:
        return json.loads(js)
    return js
def numLs():
    return str('0.1.2.3.4.5.6.7.8.9').split(',')
def ifInReturn(js,ls):
    for i in range(0,len(ls)):
        if ls[i] in js:
            js = js[ls[i]]
            return js
    return js
def ifN(resp,js):
    resp[0],resp[1] = str(resp[0]),int(0)
    if 'n' in js:
         resp[1] = int(js['n'])-1
    if str(resp[-1]) in js:
       resp[-1] = str(js[resp[-1]])
    return resp
def getResp(ls,keys,resp):
    for i in range(0,len(ls)):
        if ls[i] in keys:
            resp = resp[ls[i]]
            return resp
    return resp
def getResponse():
    rPr,resp = tallyRespSink(),fun.reader('ans.json')
    if rPr[2] == 'url':
        guiFuns.defaultOverWindow(imgScreen(urlToImg(resp)),'iimage')
    resp = json.loads(fun.reader('ans.json'))
    keys = getKeys(resp)
    for k in range(0,2):
        resp=getResp([[rPr[0],'results','choices','data'],[rPr[2],'data','text','url','response_format']][k],keys,resp)
        if fun.isLs(resp):
            keys = resp
            if int(rPr[1])-1 in range(0,len(resp)):
                resp = resp[int(rPr[1])-1]
            else:
                resp = resp[0]
    print(resp)
                
    return resp
def getMenuLs(na,ls,defa):
        return [sg.Push(),sg.T(na+':'),  sg.Combo(ls, default_value=ls[0], readonly=True, k=na,size=(30,4), background_color=sg.theme_button_color_background())]

def getSpecInfo(ls):
    info = getAllInfo('info')
    if fun.isLs(ls):
        return info[ls[0]][ls[1]]
    return info[ls]
def getAllInfo(sect):
        infos = {"map": {"category": "", "specialization": ""}, "categories": {"completions": ["chat", "translate", "qanda", "parse"], "coding": ["editcode", "debugcode", "convertcode", "writecode"],"moderation": ["moderate"], "images": ["image_create", "image_edit", "image_variation"]}, "parameters": {"all": ["all", "model", "prompt", "suffix", "max_tokens", "temperature", "top_p", "n", "stream", "logprobs", "echo", "stop", "presence_penalty", "frequency_penalty", "best_of", "logit_bias", "user", "input", "instruction", "size", "response_format", "image", "mask", "file", "purpose", "file_id", "training_file", "validation_file", "n_epochs", "batch_size", "learning_rate_multiplier", "prompt_loss_weight", "compute_classification_metrics", "classification_n_classes", "classification_positive_class", "classification_betas", "fine_tune_id", "engine_id"], "model": {"object": "str", "default": "text-davinci-003", "scale": "array", "array": ["completions", "edit", "code", "embedding"], "description": "The ID of the model to use for this request"}, "max_tokens": {"object": "int", "scale": "range", "range": {"0": 2048}, "default": 2000, "description": "The maximum number of tokens to generate in the completions.The token count of your prompt plus max_tokens cannot exceed the model context length. Most model have a context length of 2048 tokens (except for the newest model, which support 4096)."}, "logit_bias": {"object": "map", "scale": "range", "range": {"-100": 100}, "default": "None", "description": "Modify the likelihood of specified tokens appearing in the completions.Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from : 100 to 100. You can use this tokenizer tool (which works for both GPT: 2 and GPT: 3) to convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between : 1 and 1 should decrease or increase likelihood of selection; values like : 100 or 100 should result in a ban or exclusive selection of the relevant token.As an example, you can pass {50256:100} to prevent the &lt;|endoftext|&gt; token from being generated."}, "size": {"object": "str", "default": "1024x1024", "scale": "choice", "choice": ["256x256", "512x512", "1024x1024"], "description": "The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024."}, "temperature": {"object": "float", "default": 0.7, "scale": "range", "range": {"-2.0": 2.0}, "description": "What sampling temperature to use. Higher values means the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well: defined answer.We generally recommend altering this or top_p but not both."}, "best_of": {"object": "int", "default": 1, "scale": "range", "range": {"0": 10}, "description": "Generates best_of completions server: side and returns the best (the one with the highest log probability per token). Results cannot be streamed.When used with n, best_of controls the number of candidate completions and n specifies how many to return \u2013 best_of must be greater than n.Note: Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for max_tokens and stop."}, "top_p": {"object": "float", "default": 0.0, "scale": "range", "range": {"0.0": 1.0}, "description": "An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.We generally recommend altering this or temperature but not both."}, "frequency_penalty": {"object": "float", "default": 0.0, "scale": "range", "range": {"-2.0": 2.0}, "description": "Number between : 2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model likelihood to repeat the same line verbatim.See more information about frequency and presence penalties."}, "presence_penalty": {"object": "float", "default": 0.0, "scale": "range", "range": {"-2.0": 2.0}, "description": "Number between : 2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model likelihood to talk about new topics.See more information about frequency and presence penalties."}, "log_probs": {"object": "int", "default": 1, "scale": "range", "range": {"1": 10}, "description": "Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens. For example, if logprobs is 5, the API will return a list of the 5 most likely tokens. The API will always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the response.The maximum value for logprobs is 5. If you need more than this, please contact us through our Help center and describe your use case."}, "stop": {"object": "str", "default": "", "scale": "array", "range": {"0": 4}, "description": "Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence."}, "echo": {"object": "bool", "default": "False", "scale": "choice", "choice": ["True", "False"], "description": "Echo back the prompt in addition to the completions"}, "n": {"object": "int", "default": 1, "scale": "range", "range": {"1": 10}, "description": "How many completions to generate for each prompt.Note: Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for max_tokens and stop."}, "stream": {"object": "bool", "default": "False", "scale": "choice", "choice": ["True", "False"], "description": "Whether to stream back partial progress. If set, tokens will be sent as data: only server: sent events as they become available, with the stream terminated by a data: [DONE] message."}, "suffix": {"object": "str", "default": "", "scale": "range", "range": {"0": 1}, "description": "The suffix that comes after a completions of inserted text."}, "prompt": {"object": "str", "default": "None", "scale": "inherit", "description": "The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.Note that &lt;|endoftext|&gt; is the document separator that the model sees during training, so if a prompt is not specified the model will generate as if from the beginning of a new document."}, "input": {"object": "str", "default": "None", "scale": "inherit", "description": "The input text to use as a starting point for the edit."}, "instruction": {"object": "str", "default": "None", "scale": "inherit", "description": "The instruction that tells the model how to edit the prompt."}, "response_format": {"object": "str", "default": "url", "scale": "choice", "choice": ["url", "b64_json"], "description": "The format in which the generated images are returned. Must be one of url or ."}, "image": {"object": "str", "default": "None", "scale": "upload", "upload": {"type": ["PNG", "png"], "size": {"scale": {"0": 4}, "allocation": "MB"}}, "description": "The image to edit. Must be a valid PNG file, less than 4MB, and square. If mask is not provided, image must have transparency, which will be used as the mask."}, "mask": {"object": "str", "default": "None", "scale": "upload", "upload": {"type": ["PNG", "png"], "size": {"scale": {"0": 4}, "allocation": "MB"}}, "description": "An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where image should be edited. Must be a valid PNG file, less than 4MB, and have the same dimensions as image."}, "file": {"object": "str", "default": "None", "scale": "upload", "upload": {"type": ["jsonl"], "size": {"scale": {"0": 100}}, "allocation": "MB"}, "description": "Name of the JSON Lines file to be uploaded.If the purpose is set to finetune, each line is a JSON record with prompt and completions fields representing your training examples."}, "purpose": {"object": "str", "default": "None", "scale": "inherit", "description": "The intended purpose of the uploaded documents.Use finetune for Finetuning. This allows us to validate the format of the uploaded file."}, "file_id": {"object": "str", "default": "None", "scale": "inherit", "description": "The ID of the file to use for this request"}, "user": {"object": "str", "default": "defaultUser", "scale": "inherit", "description": "A unique identifier representing your end: user, which can help OpenAI to monitor and detect abuse. Learn more."}}, "descriptions": {"completions": "input what youd like to say to the bot, Have a chat with ChatGPT", "Edits": "This endpoint allows users to edit a given text prompt. It uses a generative model to suggest edits to the given prompt.", "Images": "This endpoint allows users to generate images from a given text prompt. It uses a generative model to generate an image that is similar to the given prompt.", "embeddings": "This endpoint allows users to generate embeddings from a given text prompt. It uses a generative model to generate an embedding that is similar to the given prompt.", "Files": "This endpoint allows users to upload and store files. It provides a secure way to store files in the cloud.", "Fine-Tunes": "This endpoint allows users to fine-tune a given model. It uses a generative model to fine-tune the given model to better fit the user\u2019s needs.", "Moderations": "This endpoint allows users to moderate content. It uses a generative model to detect and remove inappropriate content.", "choices": "choose from the selection", "types": "choose from the selection", "coding": "write some code", "public": "Toggle public access", "private": "Toggle private access", "help": "will display all descriptions", "temp": "pick the randomness of your interaction", "shouldBeAllGood": "below-----------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", "stillInTesting": "below-----------VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV", "similarity": "where results are ranked by relevance to a query string", "text_similarity": "Captures semantic similarity between pieces of text.", "text_search_query": "Semantic information retrieval over documents.", "text_embedding": "Get a vector representation of a given input that can be easily consumed by machine learning model", "text_insert": "insert text", "text_edit": "edit text", "search_document": "where results are ranked by relevance to a document", "search_query": "search query ", "code_edit": "specify the revisions that you are looking to make in the code", "code_search_code": "Find relevant code with a query in natural language.", "code_search_text": "text search in code", "image_edit": "[image]-main image; [mask] secondary image;[prompt]- input how you would like to have it edited", "params": "lists definitions and information about all parameters", "uploadfile": "upload a file to be used in future queries"}, "endpoints": {"engines": {"list": {"endpoint": "https://api.openai.com/v1/engines", "type": "GET"}, "retrieve": {"endpoint": "https://api.openai.com/v1/engines/{engine_id}", "type": "GET", "var": "{engine_id}"}}, "models": {"list": {"endpoint": "https://api.openai.com/v1/models", "type": "GET"}, "retrieve": {"endpoint": "https://api.openai.com/v1/models/{model}", "type": "GET", "var": "{model}"}}, "fine-tunes": {"create": {"endpoint": "https://api.openai.com/v1/fine-tunes", "type": "POST"}, "list": {"endpoint": "https://api.openai.com/v1/fine-tunes/{fine_tune_id}/events", "type": "GET", "var": "{fine_tune_id}"}, "retrieve": {"endpoint": "https://api.openai.com/v1/fine-tunes/{fine_tune_id}", "type": "GET", "var": "{fine_tune_id}"}, "cancel": {"endpoint": "https://api.openai.com/v1/fine-tunes/{fine_tune_id}/cancel", "type": "POST", "var": "{fine_tune_id}"}, "delete": {"endpoint": "https://api.openai.com/v1/models/{model}", "type": "DELETE", "var": "{model}"}}, "files": {"list": {"endpoint": "https://api.openai.com/v1/files", "type": "GET"}, "upload": {"endpoint": "https://api.openai.com/v1/files", "type": "POST"}, "delete": {"endpoint": "https://api.openai.com/v1/files/{file_id}", "type": "DELETE", "var": "{file_id}"}, "retrieveFile": {"endpoint": "https://api.openai.com/v1/files/{file_id}", "type": "GET", "var": "{file_id}"}, "retrieveContent": {"endpoint": "https://api.openai.com/v1/files/{file_id}/content", "type": "GET", "var": "{file_id}"}}, "completions": {"create": {"endpoint": "https://api.openai.com/v1/completions", "type": "POST"}}, "moderation": {"moderation": {"endpoint": "https://api.openai.com/v1/moderations", "type": "POST"}}, "edit": {"create": {"endpoint": "https://api.openai.com/v1/edits", "type": "POST"}}, "embeddings": {"create": {"endpoint": "https://api.openai.com/v1/embeddings", "type": "POST"}}, "image": {"create": {"endpoint": "https://api.openai.com/v1/images/generations", "type": "POST"}, "edit": {"endpoint": "https://api.openai.com/v1/images/edits", "type": "POST"}, "variation": {"endpoint": "https://api.openai.com/v1/images/variations", "type": "POST"}}}, "info": {"completions": {"endpoints": {"chat": "https://api.openai.com/v1/completions", "translate": "https://api.openai.com/v1/completions", "qanda": "https://api.openai.com/v1/completions", "parse": "https://api.openai.com/v1/completions", "response": ["choices", "n", "text"]}, "choices": {"model": {"default": "text-davinci-003", "list": ["text-ada-001", "text-davinci-003", "text-curie-001", "text-babbage-001"]}}, "specifications": {"chat": {"type": "completions", "refference": ["completions", "create"], "parameters": {"required": ["model", "prompt"], "optional": ["user", "n", "suffix", "max_tokens", "logit_bias", "temperature", "best_of", "top_p", "frequency_penalty", "presence_penalty", "stop", "echo"], "prompt": {"task": "chat", "structure": "", "vars": {"prompt": {"input": "what would you like to say to the bot?", "type": "str", "delimiter": ""}}}}}, "translate": {"type": "completions", "refference": ["completions", "create"], "parameters": {"required": ["model", "prompt"], "optional": ["user", "stream", "n", "suffix", "max_tokens", "logit_bias", "temperature", "best_of", "top_p", "frequency_penalty", "presence_penalty", "stop", "echo"], "prompt": {"task": "task: translate text", "structure": "languages to translate to:[languages];translate the following text:[text]", "vars": {"languages": {"input": "specify the target languages", "type": "list", "delimiter": "languages to translate to:\n"}, "text": {"input": "input the text you would like to have translated", "type": "text", "delimiter": "translate the following text:\n"}}}}}, "qanda": {"type": "completions", "refference": ["completions", "create"], "parameters": {"required": ["model", "prompt"], "optional": ["user", "stream", "n", "suffix", "max_tokens", "logit_bias", "temperature", "best_of", "top_p", "frequency_penalty", "presence_penalty", "stop", "echo"], "prompt": {"task": "task: question and answer", "structure": "[question]- input a question,question mark will auto add, [answer] - proposed answer to a question", "vars": {"question": {"input": "pose a question to have answered", "type": "str", "delimter": "Q:"}, "answer": {"input": "pose answer to a proposed question", "type": "str", "delimiter": "A:"}}}}}, "parse": {"type": "completions", "refference": ["completions", "create"], "parameters": {"required": ["model", "prompt"], "optional": ["user", "stream", "n", "suffix", "max_tokens", "logit_bias", "temperature", "best_of", "top_p", "frequency_penalty", "presence_penalty", "stop", "echo"], "prompt": {"task": "task: parse text,", "structure": " a [summary] of the [data] will be given in order to parse specific [subjects]:", "vars": {"summary": {"input": "summarize the text you would like to parse", "type": "text", "delimiter": "summary of data:\n"}, "subjects": {"input": "specific subjects you want to have parsed", "type": "list", "delimiter": "subjects:\n"}, "data": {"input": "text you would like to have parsed", "type": "text", "delimiter": "data to parse:\n"}}}}}}}, "coding": {"endpoints": {"editcode": "https://api.openai.com/v1/completions", "debugcode": "https://api.openai.com/v1/completions", "convertcode": "https://api.openai.com/v1/completions", "writecode": "https://api.openai.com/v1/completions", "response": ["choices", "n", "text"]}, "choices": {"language": {"default": "python", "list": ["Python", "Java", "C++", "JavaScript", "Go", "Julia", "R", "MATLAB", "Swift", "Prolog", "Lisp", "Haskell", "Erlang", "Scala", "Clojure", "F#", "OCaml", "Kotlin", "Dart"]}, "model": {"default": "code-davinci-002", "list": ["code-cushman-001", "text-davinci-003", "code-davinci-002"]}}, "specifications": {"writecode": {"type": "coding", "refference": ["completions", "create"], "parameters": {"required": ["model", "prompt"], "optional": ["user", "stream", "n", "suffix", "max_tokens", "logit_bias", "temperature", "best_of", "top_p", "frequency_penalty", "presence_penalty", "stop", "echo"], "prompt": {"task": "write code in [language] based off of specific [instruction]:", "structure": "[prompt]-describe the code; [language] - specify the target language", "vars": {"instruction": {"input": "describe what you are looking for, be specific", "type": "str", "delimiter": "instructuions:\n"}, "language": {"input": "which language would you like the code to be written in?", "type": "choice", "delimiter": "language:\n,"}}}}}, "editcode": {"type": "coding", "refference": ["completions", "create"], "parameters": {"required": ["model", "prompt"], "optional": ["user", "stream", "n", "suffix", "max_tokens", "logit_bias", "temperature", "best_of", "top_p", "frequency_penalty", "presence_penalty", "stop", "echo"], "prompt": {"task": "edit code", "structure": "edit [code] based off of specific [instructions]", "vars": {"instruction": {"input": "provide specific instructions on what you are looking to have edited about this code:", "type": "str", "delimiter": "instructions:\n"}, "code": {"input": "enter the code you would like to have edited:", "type": "str", "delimiter": "code:\n"}}}}}, "debugcode": {"type": "coding", "refference": ["completions", "create"], "parameters": {"required": ["model", "prompt"], "optional": ["user", "stream", "n", "suffix", "max_tokens", "logit_bias", "temperature", "best_of", "top_p", "frequency_penalty", "presence_penalty", "stop", "echo"], "prompt": {"task": "debug the code:", "structure": "debug the following code:\n", "vars": {"code": {"input": "the code you would like to have debugged", "type": "str", "delimiter": ""}}}}}, "convertcode": {"type": "coding", "refference": ["completions", "create"], "parameters": {"required": ["model", "prompt"], "optional": ["user", "stream", "n", "suffix", "max_tokens", "logit_bias", "temperature", "best_of", "top_p", "frequency_penalty", "presence_penalty", "stop", "echo"], "prompt": {"task": "convert code to another language:", "structure": "convert the following [code] to [language]", "vars": {"language": {"input": "the language you would like the code converted to:", "type": "str", "delimiter": "language:\n"}, "code": {"input": "the code you would like to have converted", "type": "str", "delimiter": "code:\n"}}}}}}}, "images": {"endpoints": {"image_create": "https://api.openai.com/v1/images/generations", "image_variation": "https://api.openai.com/v1/images/variations", "image_edit": "https://api.openai.com/v1/images/edits", "response": ["data", "n", "response_format"]}, "choices": {"response_format": {"default": "url", "list": ["url", "b64_json"]}, "size": {"default": "1024x1024", "list": ["256x256", "512x512", "1024x1024"]}}, "specifications": {"image_variation": {"type": "images", "refference": ["image", "create", "image_variation"], "parameters": {"required": ["image"], "optional": ["prompt", "size", "n", "response_format", "user", "n", "suffix", "max_tokens", "logit_bias", "temperature", "best_of", "top_p", "frequency_penalty", "presence_penalty", "stop", "echo"], "prompt": {"task": "image variation", "structure": "create a variation of the [image] based off of [instructions] if given:\n", "vars": {"instructions": {"input": "describe what you would like to have done with the image(s):", "type": "str", "delimiter": "instructions:\n"}}}}}, "image_create": {"type": "images", "refference": ["image", "create", "image_create"], "parameters": {"required": ["prompt"], "optional": ["size", "n", "response_format", "user", "suffix", "logit_bias"], "prompt": {"task": "image creation", "structure": "create an image based on the following [instructions]:\n", "vars": {"instructions": {"input": "describe the image you would like to create:", "type": "str", "delimiter": "instructions:"}}}}}, "image_edit": {"type": "images", "refference": ["image", "create", "image_edit"], "parameters": {"required": ["image", "prompt"], "optional": ["mask", "size", "n", "response_format", "user", "suffix", "max_tokens", "logit_bias"], "prompt": {"task": "image creation", "structure": "[image]-main image; [mask] secondary image;[prompt]- input how you would like to have it edited", "vars": {"instructions": {"input": "provide instructions describing what you would like to have done with the image(s):", "type": "str", "delimiter": "instructions:"}}}}}}}, "edit": {"endpoints": {"edit": "https://api.openai.com/v1/embeddings", "response": ["data", "n", "embedding"]}, "choices": {"model": {"default": "text-ada-001", "list": ["text-ada-001", "text-davinci-003", "text-curie-001", "text-babbage-001"]}}, "specifications": {"edits": {"type": "edits", "refference": ["edits", "create"], "parameters": {"required": ["model", "prompt"], "optional": ["user", "stream", "n", "max_tokens", "logit_bias", "temperature", "best_of", "top_p", "frequency_penalty", "presence_penalty", "stop"], "prompt": {"task": "edit text", "structure": "[image]-main image; [mask] secondary image;[prompt]- input how you would like to have it edited", "vars": {"instructions": {"input": "provide instructions describing what you would like to have edited:", "type": "str", "delimiter": "instructions:"}}}}}}}, "moderation": {"endpoints": {"moderate": "https://api.openai.com/v1/moderations", "response": ["results", "n", "text"]}, "choices": {"model": {"default": "None", "list": ["text-moderation-004", "davinci"]}}, "specifications": {"moderate": {"type": "moderation", "refference": ["completions", "moderation"], "parameters": {"required": ["input"], "optional": ["model"], "prompt": {"task": "moderation", "structure": "text to moderate:\n", "vars": {"input": {"input": "provide the text you would like to have moderated", "type": "text", "delimiter": "moderate the following"}}}}}}}}, "prevKeys": [], "jsList": [], "content": "application/json"}
        return infos[sect]
def getSpecification():
    return getAllInfo(category)['specifications'][specialization]
def getRefference():
    return getAllInfo('info')[category]['specifications'][specialization]['refference']
def returnParameters(category,specialization):
    return getAllInfo('info')[category]['specifications'][specialization]['parameters'] 
def tallyRespSink():
    return ifN(getAllInfo(category)['endpoints'][specialization]['response'],json.loads(js))
def getChoices():
    return getAllInfo('info')[category]['choices']
def getEndPoint():
    return getAllInfo('info')[category]['endpoints'][specialization]
def reqDelete():
  return requests.delete(ls[0], json=ls[1], headers=GETHeader())
def retrieveModel(model):
  return reqGet([model,None])
def getModels():
  pen(callJson(None,endpoints['models']['list']),'modelsList.json')
def getEngines():
  pen(callJson(None,endpoints['engines']['list']),'enginesList.json')
def getFiles():
  pen(callJson(None,endpoints['files']['list']),'filesList.json')
def mkType(ls):
    x,obj = ls
    if specializedSet['paramJs']['scale'] == 'upload':
        if ky in ['image','mask']:
            return str(x)#generate_from_masked_image(js[ky])
        return open(js[ky], "rb")
    if obj == 'float':
        return float(str(x).replace("'",'').replace("'",''))
    if obj == 'int':
        return int(str(x).split('.')[0].replace("'",'').replace("'",''))
    if obj == 'bool':
        return bool(x)
    else:
        return str(fun.eatAll(str(x),['"',"'"]))
def ifNotIntFl(js,ky):
    specializedSet['paramJs'] =getAllInfo('parameters')[ky]
    return mkType([fun.eatAll(str(js[ky]),['"',"'"]),getObj(ky)])
def simpleWindow(window):
    while True:             # Event Loop
        event, values = window.read()
        print(event, values)
        if event == 'OK':
            return values
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
            window.close()
    return event,values

def ifNotLs(ls):
    
    for i in range(0,len(ls)):
        bef == False
        if isLs(ls[i]) == False:
            lsN.append(ls[i])
            bef = True
        else:
           if len(lsA) != 0:
               lsA = []
def mkDefCats():
    lsN = []
    cats=fun.getKeys(getAllInfo('categories'))
    for i in range(0,len(cats)):
        lsN.append(button(cats[i],'category_'+cats[i],True))
    return lsN
def getLongestLen(ls):
    highest = [len(ls[0]),0]
    
    for i in range(1,len(ls)):
        if len(str(ls[i]))>highest[0]:
            highest = [len(ls[i]),i]
    return highest[0]
def mkListInp(na,k,w,w2):
    pad = [4,4]
    lsN = [],pad[0],pad[1]
    col = w/w2
    for i in range(0,k):
        for i in range(0,len(row)):
            lsC = []
            for k in range(0,len(col)):
                lsC.append(sg.Input(size=(w2), pad=(),key = 'name'))
        LSn.append([sg.Input(size=(), pad=(4,4),key = 'name')])
  

def getObj(x):
    return getAllInfo('parameters')[x]['object']


def tab(i):
   return [[sg.Text(specializedSet['inputTabKeys']['names'][i])],sg.Multiline(size=(100,10), font='Tahoma 13', autoscroll=True,key=str(specializedSet['inputTabKeys']['names'][i])),sg.VerticalSeparator(pad=None)]
#callopenAi()
def getDefMenu():
    return [sg.Menu([['File', ['Open', 'Save', 'Exit',]],['Edit', ['Paste', ['Special', 'Normal',], 'Undo'],],['Help', 'About...'],])]
def getDefButtons():
    return [sg.OK('OK'),sg.Button('Info'),sg.Button('Run'),sg.Button('Auto'),sg.Button('Exit')]
def getDefaultSetOptions():
    return sg.set_options(suppress_raise_key_errors=bool(False), suppress_error_popups=bool(False), suppress_key_guessing=bool(False))
def getDefaults():
    return [getDefMenu(),getDefButtons()]
def getDefaultLayout(sg1):
    lsA = getDefaults()
    lsA.append(sg1)
    return lsA

def defaultWindow2(sg1,title):
    gogo = True
    getDefaultSetOptions()
    layout = [[getDefaultLayout(sg1)]]
    window = sg.Window(title,layout , finalize=False)
    while gogo == True:
      vals = eventCall(window)
      if vals != False:
        return vals
def createLayout(ls,name):
    lsA =[]
    for i in range(0,len(ls)):
      lsA.append()
    return lsA
def searchVals(x,k,values):
    keys = getKeys(values)[1:]
    for i in range(0,len(keys)):
        print(keys[i],x)
        if fun.isNum(keys) == False:
            if x in keys[i]:
                return keys[i].split(x)[k] 
    return None
def cleanLs(ls):
    lsN = []
    for i in range(0,len(ls)):
         if ls[i] != '':
            lsN.append(ls[i])
    return lsN
def frontToBackCompare(lsN,x,k,i,y):
    c = 1
    while k <i-c:
        if x[k:i-c] in y:
        
            lsN.append(x[k:i-c])
        c +=1
    c = 0
    while k+c <i:
        if x[k+1:i] in y:
            lsN.append(x[k+1:i])
        c +=1
    return lsN
def mulLsVars(ls,k):
    ls = fun.mkLs(ls)
    for i in range(0,k):
        if fun.isLs(ls[0]):
            for k in range(0,len(ls[0])):
                ls.append(ls[i][k])
        else:
            ls.append(ls[i])
    return ls
def lenLs(ls):
    lsN = []
    for i in range(0,len(ls)):
        lsN.append(len(str(ls[i])))
        if [fun.isNum(ls[i]),isInt(ls[i]),isFloat(ls[i])]==mulLsVars([False],3):
            lsN[i] = int(ls[i])
    return lsN
def isInt(x):
    if type(x) is int:
        return True
    return False
def isFloat(x):
    if type(x) is float:
        return True
    return False
def getLongestLen(ls):
    lsN,lsF = lenLs(ls),[]
    lsL = lsN
    lsL.sort()
    for i in range(0,len(lsN)):
        if len(ls[i]) == lsL[-1] and ls[i] not in lsF:
            lsF.append(ls[i])
    return lsF
def compareStr(x,y):
    ls=lowerSrts(str(x),str(y))
    lsN=[]
    if ls[0] == ls[1]:
        return ls[1]
    for i in range(0,2):
        if ls[1-i] in ls[0-i]:
            ls[0-i] = float(float(len(ls[0-i]))/float(len(ls[1-i])))
    k = 0
    for i in range(0,len(str(ls[0]))):
        if ls[0][k:i] not in ls[1]:
            lsN = frontToBackCompare(lsN,ls[0],k,i,ls[1])
    return getLongestLen(cleanLs(frontToBackCompare(lsN,x,k,i,y)))
def ifNotInLsApp(x,ls):
    if x not in ls:
        ls.append(x)
    return ls
def lowerSrts(x,y):
    return str(x).lower(),str(y).lower()
def ifInRet2(na,dic):
    keys = getKeys(dic)
    if na in dic:
        return na,dic[na]
    for i in range(0,len(keys)):
        names =fun.mkLs(compareStr(na,keys[i]))
        for k in range(0,len(names)):
            if str(names[k]) in dic:
                return names[k],dic[names[k]]
    input([na,keys])
    return na,dic
def countIt(x,y):
    cou = 0
    if str(y) in str(x):
        cou = (len(x)-len(str(x).replace(str(y),'')))/len(y)
    return cou
def countItLs(ls,x):
    cou = 0
    for i in range(0,len(ls)):
        if x == ls[i]:
            cou +=1
    return cou
def isInLsStr(ls,x):
    for i in range(0,len(ls)):
        if x in ls[i]:
            return True
    return False
def isInLsStrInStrClear(ls,x):
    lsN = []
    for i in range(0,len(ls)):
        if ls[i] not in x:
            lsN.append(ls[i])
    return lsN
def isInLsStrRet(ls,x):
    lsN= []
    for i in range(0,len(ls)):
        if x in ls[i]:
            lsN.append(ls[i])
    return lsN
def findItI(ls,x):
    for i in range(0,len(ls)):
        if x == ls[i]:
            return i
    return False
def getStrLenOfLs(ls):
    length = 0
    for k in range(0,len(ls)):
        length+=len(ls[k])
    return length
def ifInRet2(na,dic):
    keys = getKeys(dic)
    if na in dic:
        return na,dic[na]
    for i in range(0,len(keys)):
        if str(na).lower() == str(keys[i]).lower():
            return keys[i],dict[keys[i]]
    lsAll = []
    for c in range(0,len(keys)):
        lsN = []
        for i in range(0,len(na)):
            n = na[i:]
            for k in range(0,len(n)):
                if n[i:k] in keys[c]:
                    if isInLsStr(lsN,n[i:k]) == False:
                        lsN = isInLsStrInStrClear(lsN,n[i:k])
                        lsN.append(n[i:k])
                        
        lsAll.append(cleanLs(lsN))
    highest = [getStrLenOfLs(lsAll[i]),0]
    for i in range(1,len(lsAll)):
        length = getStrLenOfLs(lsAll[i])
        if length > highest[0]:
            highest = [length,i]
    if fun.isLs(dic):
        return keys[highest[1]],dic
    return keys[highest[1]],dic[keys[highest[1]]]
def lsNotInLs(ls,ls2):
    lsN=[]
    for i in range(0,len(ls)):
        if ls[i] not in ls2:
            lsN.append(ls[i])
    return lsN
def removeFromLs(ls,x):
    lsN=[]
    for i in range(0,len(ls)):
        if ls[i] != x:
            lsN.append(ls[i])
    return lsN
def checkTabCreate(window):
    global tabIndex
    js = getAllThings()
    varKeys,jsVars,lsN = getKeys(js['prompt']['parse']['vars']),js['prompt']['parse']['vars'],[]
    oldTabs = lsNotInLs(tabIndex,varKeys)
    
    for i in range(0,len(oldTabs)):
        tabIndex = removeFromLs(tabIndex,oldTabs[i])
        window[oldTabs[i]+'_Tab'].update(visible=False)
        window[oldTabs[i]+'_Tab'].update(disabled=True)

    for k in range(0,len(varKeys)):
        varKey = varKeys[k]
        newTab = getTab(varKey,[[txtBox(jsVars[varKey][getKeys(jsVars[varKey])[0]],varKey,None,None,True,False)],txtInputs(varKey,varKey,(50,10),'Tahoma 13',True,False,None)],varKey+'_Tab',True)
        if len(varKeys) > len(tabIndex):
            if varKey not in tabIndex:
                lsN.append(newTab)
                if window != None:
                    window['tabGroupInput'].add_tab(newTab)
                tabIndex.append(varKey)
    return lsN   
def getInfoSec(na,part):
    if part != False:
        if na not in part:
           
            na,part = ifInRet2(na,part)
    return {'names':na,'parse':part[na]}
def getAllThings():
    js ={'category':getInfoSec(category,info),
         'specifications':getInfoSec('specifications',info[category]),
         'specialization':getInfoSec(specialization,info[category]['specifications']),
         'parameters':getInfoSec('parameters',info[category]['specifications'][specialization])
         }
    js['prompt'] = getInfoSec('prompt',js['parameters']['parse'])
    js['structure'] = getInfoSec('structure',js['prompt']['parse'])
    js['categoryDefinition']=ifInRet2(category,descriptions)[1]
    js['specializationDeffinition']=ifInRet2(specialization,descriptions)[1]
    
    return js
def whileWindow(window):
    while True:
        event, values = window.read()
        print(values)
        if start == True:
            fun.pen(json.dumps({'defaul':values}),'valuesJs.json')
            start = False
        if event == 'Compile':
            specializedSet['jsList'] = compilePrompt(values)
        if event == 'SEND':
            print(reqPost(specializedSet['jsList']).text)
            query = value['query'].rstrip()
            print('The command you entered was {}'.format(query))
            command_history.append(query)
            history_offset = len(command_history)-1
            window['query'].update('')
            window['history'].update('\n'.join(command_history[-3:]))
        elif event in (sg.WIN_CLOSED, 'EXIT'):
            break
        elif 'Up' in event and len(command_history):
            command = command_history[history_offset]
            # decrement is not zero
            history_offset -= 1 * (history_offset > 0)
            window['query'].update(command)
        elif 'Down' in event and len(command_history):
            history_offset += 1 * (history_offset < len(command_history)-1)
            command = command_history[history_offset]
            window['query'].update(command)
        elif 'Escape' in event:
            window['query'].update('')
        elif event == sg.WIN_CLOSED or event == 'Exit':
            break
            window.close()
        elif 'default_' in event:
            na,defa = event.split('default_')[0],event.split('default_')[1]
            if values[event] == True:
                defa = str(defa).replace("'",'').replace('"','')
                if str(defa) in ['True','False','None']:
                    window[na].update(value=bool(defa))
                elif '.' in str(defa):
                    window[na].update(value=float(defa))
                elif fun.isNum(defa):
                    window[na].update(value=int(defa))
                else:
                    window[na].update(value=str(defa))   
        elif '_info' in event:
            req = 'optional'
            na = event.split('_info')[0]
            defa = searchVals(str(na)+'default_',-1,values)
            if str(na)+str('disable') in values:
                if values[str(na)+str('disable')] == True:
                    req = 'required'
            from dataSheets import parameters
            print(parameters)
            sg.popup_scrolled('na,  | default:'+str(defa)+' | '+str(req)+'\n',parameters[na]['description'],)
            getAllThings()
        elif 'disable' in event:
            print(event)
            na = event.split('disable')[0]
            if values[na] == True:
                keys = getKeys(values)[1:]
                for i in range(0,len(keys)):
                    print(keys[i],na)
                    if fun.isNum(keys) == False:
                        if na+'default_' in keys[i]:
                            defa = keys[i].split(na+'default_')[-1]
                            print(keys[i])
                            window[na].update(value=defa)
        elif event == 'Edit Me':
              sg.execute_editor(__file__)
        elif event == 'Version':
              sg.popup_scrolled(sg.get_versions(), keep_on_top=True)
        elif event == 'File Location':
              sg.popup_scrolled('This Python file is:', __file__)
        elif 'category_' in event:
            js = getAllThings()
            ls = categoriesJs[event.split('category_')[1]]
            changeGlob('category',event.split('category_')[1])
            changeGlob('specializedLs',categoriesJs[category])
            changeGlob('specialization',specializedLs[0])
            window['structure'].update(value=js['structure']['parse'])
            window['categoryDisplay'].update(value=js['category']['names'])
            window['categoryDescription'].update(value=js['categoryDefinition'])
            window['specializationDisplay'].update(value=js['specialization']['names'])
            window['specializationDescription'].update(value=js['specializationDeffinition'])
            window["catCombo"].update(values = ls)
            window["catCombo"].update(value=ls[0])
            varKeys = getKeys(js['prompt']['parse']['vars'])
            checkTabCreate(window)
            #layout = [[getDefMenu()],[getBanner()],[getTop()],[getquerySection()],[getPrevQuery()],[getOutput()],[getInput()]]

            window.refresh()
        elif event == 'Generate':
            specializedSet['jsList'] = [[sg.Text('Parameters', font='Any 20')],[txtBox('def  '),txtBox('dis  '),txtBox('   inf  ')]]
            getParamDrop([category,changeGlob('specialization',values['catCombo'])])
            window.refresh()
            window.close()
            break
           
            return 
                
    return values
global category,specializations,categories,specialization,catKeys,specializedSet,catLs,categoriesJs,specializations,infotabIndex
from dataSheets import descriptions,specifications
mid={'completions':'create','images':'create','Embedding':'create','coding':'create'}
categories={"completions":["chat","translate","qanda","parse"],"coding":["editcode","debugcode","convertcode","writecode"],"moderation":["moderate"],"images":["image_create","image_edit","image_variatoin"]}
parameters={"all":['all','model', 'prompt', 'suffix', 'max_tokens', 'temperature', 'top_p', 'n', 'stream', 'logprobs', 'echo', 'stop', 'presence_penalty', 'frequency_penalty', 'best_of', 'logit_bias', 'user', 'input', 'instruction', 'size', 'response_format', 'image', 'mask', 'file', 'purpose', 'file_id', 'training_file', 'validation_file', 'n_epochs', 'batch_size', 'learning_rate_multiplier', 'prompt_loss_weight', 'compute_classification_metrics', 'classification_n_classes', 'classification_positive_class', 'classification_betas', 'fine_tune_id', 'engine_id'],"model": {"object": "str", "default": "text-davinci-003", "scale": "array", "array": ["completions", "edit", "code", "Embedding"], "description": "The ID of the model to use for this request"}, "max_tokens": {"object": "int", "scale": "range", "range": {0: 2048}, "default": 2000, "description": "The maximum number of tokens to generate in the completions.The token count of your prompt plus max_tokens cannot exceed the model context length. Most model have a context length of 2048 tokens (except for the newest model, which support 4096)."}, "logit_bias": {"object": "map", "scale": "range", "range": {-100: 100}, "default": "None", "description": "Modify the likelihood of specified tokens appearing in the completions.Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from : 100 to 100. You can use this tokenizer tool (which works for both GPT: 2 and GPT: 3) to convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between : 1 and 1 should decrease or increase likelihood of selection; values like : 100 or 100 should result in a ban or exclusive selection of the relevant token.As an example, you can pass {50256:100} to prevent the &lt;|endoftext|&gt; token from being generated."}, "size": {"object": "str", "default": "1024x1024", "scale": "choice", "choice": ["256x256", "512x512", "1024x1024"], "description": "The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024."}, "temperature": {"object": "float", "default": 0.7, "scale": "range", "range": {-2.0: 2.0}, "description": "What sampling temperature to use. Higher values means the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well: defined answer.We generally recommend altering this or top_p but not both."}, "best_of": {"object": "int", "default": 1, "scale": "range", "range": {0: 10}, "description": "Generates best_of completions server: side and returns the best (the one with the highest log probability per token). Results cannot be streamed.When used with n, best_of controls the number of candidate completions and n specifies how many to return – best_of must be greater than n.Note: Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for max_tokens and stop."}, "top_p": {"object": "float", "default": 0.0, "scale": "range", "range": {0.0: 1.0}, "description": "An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.We generally recommend altering this or temperature but not both."}, "frequency_penalty": {"object": "float", "default": 0.0, "scale": "range", "range": {-2.0: 2.0}, "description": "Number between : 2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model likelihood to repeat the same line verbatim.See more information about frequency and presence penalties."}, "presence_penalty": {"object": "float", "default": 0.0, "scale": "range", "range": {-2.0: 2.0}, "description": "Number between : 2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model likelihood to talk about new topics.See more information about frequency and presence penalties."}, "log_probs": {"object": "int", "default": 1, "scale": "range", "range": {1: 10}, "description": "Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens. For example, if logprobs is 5, the API will return a list of the 5 most likely tokens. The API will always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the response.The maximum value for logprobs is 5. If you need more than this, please contact us through our Help center and describe your use case."}, "stop": {"object": "str", "default": "", "scale": "array", "range": {0: 4}, "description": "Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence."}, "echo": {"object": "bool", "default": "False", "scale": "choice", "choice": ["True", "False"], "description": "Echo back the prompt in addition to the completions"}, "n": {"object": "int", "default": 1, "scale": "range", "range": {1: 10}, "description": "How many completions to generate for each prompt.Note: Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for max_tokens and stop."}, "stream": {"object": "bool", "default": "False", "scale": "choice", "choice": ["True", "False"], "description": "Whether to stream back partial progress. If set, tokens will be sent as data: only server: sent events as they become available, with the stream terminated by a data: [DONE] message."}, "suffix": {"object": "str", "default": "", "scale": "range", "range": {0: 1}, "description": "The suffix that comes after a completions of inserted text."}, "prompt": {"object": "str", "default": "None", "scale": "inherit", "description": "The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.Note that &lt;|endoftext|&gt; is the document separator that the model sees during training, so if a prompt is not specified the model will generate as if from the beginning of a new document."}, "input": {"object": "str", "default": "None", "scale": "inherit", "description": "The input text to use as a starting point for the edit."}, "instruction": {"object": "str", "default": "None", "scale": "inherit", "description": "The instruction that tells the model how to edit the prompt."}, "response_format": {"object": "str", "default": "url", "scale": "choice", "choice": ["url", "b64_json"], "description": "The format in which the generated images are returned. Must be one of url or ."}, "image": {"object": "str", "default": "None", "scale": "upload", "upload": {"type": ["PNG", "png"], "size": {"scale": {0: 4}, "allocation": "MB"}}, "description": "The image to edit. Must be a valid PNG file, less than 4MB, and square. If mask is not provided, image must have transparency, which will be used as the mask."}, "mask": {"object": "str", "default": "None", "scale": "upload", "upload": {"type": ["PNG", "png"], "size": {"scale": {0: 4}, "allocation": "MB"}}, "description": "An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where image should be edited. Must be a valid PNG file, less than 4MB, and have the same dimensions as image."}, "file": {"object": "str", "default": "None", "scale": "upload", "upload": {"type": ["jsonl"], "size": {"scale": {0: 100}}, "allocation": "MB"}, "description": "Name of the JSON Lines file to be uploaded.If the purpose is set to finetune, each line is a JSON record with prompt and completions fields representing your training examples."}, "purpose": {"object": "str", "default": "None", "scale": "inherit", "description": "The intended purpose of the uploaded documents.Use finetune for Finetuning. This allows us to validate the format of the uploaded file."}, "file_id": {"object": "str", "default": "None", "scale": "inherit", "description": "The ID of the file to use for this request"}, "user": {"object": "str", "default": "defaultUser", "scale": "inherit", "description": "A unique identifier representing your end: user, which can help OpenAI to monitor and detect abuse. Learn more."}}
specifications={'completions':{"create":{'type': 'completions', 'delims': ['', ''], 'model': {'default': 'text-davinci-003', 'choices': ['text-ada-001', 'text-davinci-003', 'text-curie-001', 'text-babbage-001']}, 'clients': "@mulChoice(specifications[typ]['model']['choices'],'model')\n@mulChoice(['True','False'],'stream')", 'defaults': {'model': 'text-davinci-003', 'prompt': 'None', 'user': 'defaultUser', 'stream': 'False', 'n': '1', 'suffix': 'None', 'max_tokens': '2000', 'temperature': '0.7', 'best_of': '1', 'top_p': '0.0', 'frequency_penalty': '0.0', 'presence_penalty': '0.0', 'stop': 'None', 'echo': 'False'}}},'coding': {'type': 'coding', 'delims': ['', ''], 'model': {}, 'clients': "@mulChoice(['Python','Java','C++','JavaScript','Go','Julia','R','MATLAB','Swift','Prolog','Lisp','Haskell','Erlang','Scala','Clojure','F#','OCaml','Kotlin','Dart'],'language')\n@mulChoice(specifications[typ]['model']['choices'],'model')\n\n@mulChoice(['True','False'],'echo')\n@mulChoice(['True','False'],'stream')"}, 'Embedding': {'type': 'Embedding', 'delims': ['', ''], 'model': {'default': 'text-Embedding-ada-002', 'choices': ['text-ada-001', 'text-davinci-003', 'text-curie-001', 'text-babbage-001']}, 'clients': "@mulChoice(specifications[typ]['model']['choices'],'model')"}, 'moderations': {'type': 'moderation', 'delims': ['', ''], 'model': {'default': 'text-davinci-003', 'model': {'default': 'text-davinci-003', 'choices': ['text-davinci-003', 'text-moderation-001']}, 'clients': "@mulChoice(specifications[typ]['model']['choices'],'model')\n"}, 'prompt': '[input] - input text you would like to have moderated'}, 'edits': {'type': 'edits', 'delims': ['', ''], 'model': {'default': 'text-ada-001', 'choices': ['text-ada-001', 'text-davinci-003', 'text-curie-001', 'text-babbage-001']}, 'clients': "@mulChoice(specifications[typ]['model']['choices'],'model')\n@mulChoice(['True','False'],'echo')\n@mulChoice(['True','False'],'stream')", 'prompt': '[input]-enter your text; [instruction]- tell it what you want it to do.'}, 'images': {'type': 'images', 'delims': ['', ''], 'model': {}, 'clients': "\t@imageSize()\n@mulChoice(['url','b64_json'])"}, 'Embedding': {'type': 'Embedding', 'defaults': {'model': 'text-davinci-003', 'user': 'defaultUser', 'input': 'None'}}, 'image_variation': {'type': 'images', 'delims': ['', ''], 'model': {}, 'clients': "\t@imageSize()\n@mulChoice(['url','b64_json'])", 'prompt': '[image]- upload an image of your choice; [prompt]- input how you would like it edited', 'defaults': {'image': 'None', 'prompt': 'None', 'user': 'defaultUser', 'n': '1', 'size': '1024x1024', 'response_format': 'url'}}, 'image_create': {'type': 'images', 'delims': ['', ''], 'model': {}, 'clients': "\t@imageSize()\n@mulChoice(['url','b64_json'])", 'prompt': '[prompt]- input what image you would like to have formulated', 'defaults': {'prompt': 'None', 'user': 'defaultUser', 'n': '1', 'size': '1024x1024', 'response_format': 'url', 'image': 'None', 'mask': 'None'}}, 'image_edit': {'type': 'images', 'delims': ['', ''], 'model': {}, 'clients': "\t@imageSize()\n@mulChoice(['url','b64_json'])", 'prompt': '[image]-main image; [mask] secondary image;[prompt]- input how you would like to have it edited', 'defaults': {'prompt': 'None', 'user': 'defaultUser', 'n': '1', 'size': '1024x1024', 'response_format': 'url'}}, 'moderation': {'type': 'moderate', 'delims': ['moderate this text:', ''], 'model': {'default': 'text-moderation-003', 'choices': ['text-moderation-003', 'text-moderation-001']}, 'clients': "@mulChoice(specifications[typ]['model']['choices'],'model')\n\t", 'defaults': {'moderate': {'model': 'text-davinci-003', 'user': 'defaultUser', 'input': 'None'}}}, 'edit': {'type': 'edit', 'delims': ['', ''], 'model': {'default': 'text-davinci-edit-001', 'clients': ['text-davinci-edit-001']}, 'clients': "@mulChoice(specifications[typ]['model']['choices'],'model')\n\t", 'defaults': {'image': 'file', 'mask': 'file', 'model': 'text-davinci-003', 'user': 'defaultUser', 'n': '1', 'temperature': '0.7', 'top_p': '0.0', 'input': 'None', 'instruction': 'None'}}, 'uploadfile': {'type': 'completions', 'delims': ['file', 'name']}, 'translate': {'type': 'completions', 'delims': ['#i will need for you to translate [text] into [languages]:', 'languages:', 'text:'], 'vars': ['languages', 'text'], 'prompt': '[prompt] - enter the text you would like to translate;[language] -enter the desired languages'}, 'qanda': {'type': 'completions', 'delims': ['Q:', 'A:'], 'vars': ['question'], 'prompt': '[prompt]- input a question,question mark will auto add'}, 'chat': {'type': 'completions', 'delims': ['', ''], 'vars': ['prompt'], 'prompt': '[prompt] - input what youd like to say to the bot, Have a chat with ChatGPT'}, 'parse': {'type': 'completions', 'delims': ['#this query is for parsing, a [summary] of the [data] will be given in order to parse specific [variables]:', 'summary:', 'data:', 'variables:'], 'vars': ['summary', 'data', 'variables'], 'prompt': '[summerize]-summarize the text;[subjects]-comma seperated subjects to parse;[prompt]-entertext'},'writecode': {'type': 'coding', 'delims': ['#write code in [language] based off of specific [instruction]:', 'language:', 'instruction'], 'vars': ['language', 'instruction'], 'prompt': '[prompt]-describe the code; [language] - specify the target language'},'editcode': {'type': 'coding', 'delims': ['#edit based off of specific [instructions] i will need you to write [code]:', 'instructon:', 'code:'], 'vars': ['instructon', 'code'], 'prompt': '[prompt]-describe what your focus is;[code]- enter your code'},'debugcode': {'type': 'coding', 'delims': ['#debug [code] based off of specific [instructions]:', 'code:', 'instructions:'], 'vars': ['code', 'instructions'], 'prompt': '[prompt]-describe what your focus is;[code]- enter your code'},'convertcode': {'type': 'coding', 'delims': ['#convert [code] to [language]:', 'code:', 'language:'], 'vars': ['code', 'language'], 'prompt': '[code]-input your code;[language]-input the language youd like to convert to'}, 'text_search_doc': {'type': 'Embedding', 'delims': ['', '']}, 'similarity': {'type': 'Embedding', 'delims': ['', '']}, 'text_similarity': {'type': 'Embedding', 'delims': ['', '']}, 'text_search_query': {'type': 'Embedding', 'delims': ['', '']}, 'text_Embedding': {'type': 'Embedding', 'delims': ['', '']}, 'text_insert': {'type': 'Embedding', 'delims': ['', '']}, 'text_edit': {'type': 'Embedding', 'delims': ['', '']}, 'search_document': {'type': 'Embedding', 'delims': ['', '']}, 'search_query': {'type': 'Embedding', 'delims': ['', '']}, 'instruct': {'type': 'Embedding', 'delims': ['', '']}, 'code_edit': {'type': 'Embedding', 'delims': ['', '']}, 'code_search_code': {'type': 'Embedding', 'delims': ['', '']}, 'code_search_text': {'type': 'Embedding', 'delims': ['', '']}}
choi = ["translate","qanda","chat","parse","mention","writecode","uploadcode","editcode","debugcode","convertcode","image_create","image_edit","image_variation","text_search_doc","similarity","text_similarity","text_search_query","text_Embedding","text_insert","text_edit","search_document","search_query","instruct","code_edit","code_search_code","code_search_text","moderation","edit","private","public","help","params",'uploadfile']            
descriptions= {'completions':'allows users to generate a completions of a given text prompt. It uses a generative model to generate a response that is similar to the given prompt.','chat':'have a with an open ai model.', 'Edits': 'This endpoint allows users to edit a given text prompt. It uses a generative model to suggest edits to the given prompt.', 'Images': 'This endpoint allows users to generate images from a given text prompt. It uses a generative model to generate an image that is similar to the given prompt.', 'Embedding': 'This endpoint allows users to generate Embedding from a given text prompt. It uses a generative model to generate an Embedding that is similar to the given prompt.', 'Files': 'This endpoint allows users to upload and store files. It provides a secure way to store files in the cloud.', 'Fine-Tunes': 'This endpoint allows users to fine-tune a given model. It uses a generative model to fine-tune the given model to better fit the user’s needs.', 'Moderations': 'This endpoint allows users to moderate content. It uses a generative model to detect and remove inappropriate content.', 'completions': 'input what youd like to say to the bot, Have a chat with ChatGPT', 'choices': 'choose from the selection', 'types': 'choose from the selection', 'coding': 'write some code', 'public': 'Toggle public access', 'private': 'Toggle private access', 'help': 'will display all descriptions', 'temp': 'pick the randomness of your interaction', 'shouldBeAllGood': 'below-----------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', 'stillInTesting': 'below-----------VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV', 'similarity': 'where results are ranked by relevance to a query string', 'text_similarity': 'Captures semantic similarity between pieces of text.', 'text_search_query': 'Semantic information retrieval over documents.', 'text_Embedding': 'Get a vector representation of a given input that can be easily consumed by machine learning model', 'text_insert': 'insert text', 'text_edit': 'edit text', 'search_document': 'where results are ranked by relevance to a document', 'search_query': 'search query ', 'code_edit': 'specify the revisions that you are looking to make in the code', 'code_search_code': 'Find relevant code with a query in natural language.', 'code_search_text': 'text search in code', 'image_edit': '[image]-main image; [mask] secondary image;[prompt]- input how you would like to have it edited', 'params': 'lists definitions and information about all parameters', 'uploadfile': 'upload a file to be used in future queries'}
endpoints = {"engines": {"list": {"endpoint": "https://api.openai.com/v1/engines", "type": "GET"}, "retrieve": {"endpoint": "https://api.openai.com/v1/engines/{engine_id}", "type": "GET", "var": "{engine_id}"}}, "models": {"list": {"endpoint": "https://api.openai.com/v1/models", "type": "GET"}, "retrieve": {"endpoint": "https://api.openai.com/v1/models/{model}", "type": "GET", "var": "{model}"}}, "moderation": {"create": "https://api.openai.com/v1/moderations", "type": "POST"}, "fine-tunes": {"create": {"endpoint": "https://api.openai.com/v1/fine-tunes", "type": "POST"}, "list": {"endpoint": "https://api.openai.com/v1/fine-tunes/{fine_tune_id}/events", "type": "GET", "var": "{fine_tune_id}"}, "retrieve": {"endpoint": "https://api.openai.com/v1/fine-tunes/{fine_tune_id}", "type": "GET", "var": "{fine_tune_id}"}, "cancel": {"endpoint": "https://api.openai.com/v1/fine-tunes/{fine_tune_id}/cancel", "type": "POST", "var": "{fine_tune_id}"}, "delete": {"endpoint": "https://api.openai.com/v1/models/{model}", "type": "DELETE", "var": "{model}"}}, "completions": {"create": {"endpoint": "https://api.openai.com/v1/completions", "type": "POST"}}, "edit": {"create": {"endpoint": "https://api.openai.com/v1/edits", "type": "POST"}}, "files": {"list": {"endpoint": "https://api.openai.com/v1/files", "type": "GET"}, "upload": {"endpoint": "https://api.openai.com/v1/files", "type": "POST"}, "delete": {"endpoint": "https://api.openai.com/v1/files/{file_id}", "type": "DELETE", "var": "{file_id}"}, "retrieveFile": {"endpoint": "https://api.openai.com/v1/files/{file_id}", "type": "GET", "var": "{file_id}"}, "retrieveContent": {"endpoint": "https://api.openai.com/v1/files/{file_id}/content", "type": "GET", "var": "{file_id}"}}, "Embedding": {"create": {"endpoint": "https://api.openai.com/v1/Embedding", "type": "POST"}}, "image": {"create": {"endpoint": "https://api.openai.com/v1/images/generations", "type": "POST"}, "edit": {"endpoint": "https://api.openai.com/v1/images/edits", "type": "POST"}, "variation": {"endpoint": "https://api.openai.com/v1/images/variations", "type": "POST"}}}
paramNeeds = {
    'completions':{'create':{'required':['model','prompt'],'optional':['user','stream','n','suffix','max_tokens','logit_bias','temperature','best_of','top_p','frequency_penalty','presence_penalty','stop','echo'],'models':{'default':'text-davinci-003','choices':['text-ada-001','text-davinci-003','text-curie-001','text-babbage-001']}}},
    'codeing':{'create':{'required':['model','prompt'],'optional':['user','stream','n','suffix','max_tokens','logit_bias','temperature','best_of','top_p','frequency_penalty','presence_penalty','stop','echo'],'models':{'default':'text-davinci-003','choices':['code-cushman-001', 'code-davinci-002']}}},
    'edit':{'create':{'required':['model','instruction'],'optional':['user','input','n','temperature','top_p'],'models':{'default':"text-davinci-edit-001",'choices':["text-davinci-edit-001"]}}},
    'image':{'create':{'required':['prompt'],'optional':['user','size','n','response_format']},
                       'edit':{'required':['image','prompt'],'optional':['mask','user','size','n','response_format']},
                       'variation':{'required':['image'],'optional':['prompt','mask','user','size','n','response_format']}},
    'Embedding':{'create':{'required':['model','input'],'optional':['user'],'models':{'default':'text-Embedding-ada-002','choices':['text-ada-001','text-davinci-003','text-curie-001','text-babbage-001']}}},
    'moderation':{'moderate':{'required':['input'],'optional':['model','user'], 'models':{'default':"text-moderation-001",'choices':["text-moderation-001"]}}}}
models = {"object": "list", "data": [{"id": "babbage", "object": "model", "created": 1649358449, "owned_by": "openai", "permission": [{"id": "modelperm-49FUp5v084tBB49tC4z8LPH5", "object": "model_permission", "created": 1669085501, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "babbage", "parent": "None"}, {"id": "ada", "object": "model", "created": 1649357491, "owned_by": "openai", "permission": [{"id": "modelperm-xTOEYvDZGN7UDnQ65VpzRRHz", "object": "model_permission", "created": 1669087301, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "ada", "parent": "None"}, {"id": "davinci", "object": "model", "created": 1649359874, "owned_by": "openai", "permission": [{"id": "modelperm-U6ZwlyAd0LyMk4rcMdz33Yc3", "object": "model_permission", "created": 1669066355, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "davinci", "parent": "None"}, {"id": "text-Embedding-ada-002", "object": "model", "created": 1671217299, "owned_by": "openai-internal", "permission": [{"id": "modelperm-Ad4J5NsqPbNJy0CMGNezXaeo", "object": "model_permission", "created": 1672848112, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-Embedding-ada-002", "parent": "None"}, {"id": "babbage-code-search-code", "object": "model", "created": 1651172509, "owned_by": "openai-dev", "permission": [{"id": "modelperm-4qRnA3Hj8HIJbgo0cGbcmErn", "object": "model_permission", "created": 1669085863, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "babbage-code-search-code", "parent": "None"}, {"id": "text-similarity-babbage-001", "object": "model", "created": 1651172505, "owned_by": "openai-dev", "permission": [{"id": "modelperm-48kcCHhfzvnfY84OtJf5m8Cz", "object": "model_permission", "created": 1669081947, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-similarity-babbage-001", "parent": "None"}, {"id": "text-davinci-001", "object": "model", "created": 1649364042, "owned_by": "openai", "permission": [{"id": "modelperm-MVM5NfoRjXkDve3uQW3YZDDt", "object": "model_permission", "created": 1669066355, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-davinci-001", "parent": "None"}, {"id": "curie-instruct-beta", "object": "model", "created": 1649364042, "owned_by": "openai", "permission": [{"id": "modelperm-JlSyMbxXeFm42SDjN0wTD26Y", "object": "model_permission", "created": 1669070162, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "curie-instruct-beta", "parent": "None"}, {"id": "babbage-code-search-text", "object": "model", "created": 1651172509, "owned_by": "openai-dev", "permission": [{"id": "modelperm-Lftf8H4ZPDxNxVs0hHPJBUoe", "object": "model_permission", "created": 1669085863, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "babbage-code-search-text", "parent": "None"}, {"id": "babbage-similarity", "object": "model", "created": 1651172505, "owned_by": "openai-dev", "permission": [{"id": "modelperm-mS20lnPqhebTaFPrcCufyg7m", "object": "model_permission", "created": 1669081947, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "babbage-similarity", "parent": "None"}, {"id": "curie-search-query", "object": "model", "created": 1651172509, "owned_by": "openai-dev", "permission": [{"id": "modelperm-O30H5MRAHribJNyy87ugfPWF", "object": "model_permission", "created": 1669066354, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "curie-search-query", "parent": "None"}, {"id": "code-search-babbage-text-001", "object": "model", "created": 1651172507, "owned_by": "openai-dev", "permission": [{"id": "modelperm-EC5ASz4NLChtEV1Cwkmrwm57", "object": "model_permission", "created": 1669085863, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "code-search-babbage-text-001", "parent": "None"}, {"id": "code-cushman-001", "object": "model", "created": 1656081837, "owned_by": "openai", "permission": [{"id": "modelperm-M6pwNXr8UmY3mqdUEe4VFXdY", "object": "model_permission", "created": 1669066355, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "code-cushman-001", "parent": "None"}, {"id": "code-search-babbage-code-001", "object": "model", "created": 1651172507, "owned_by": "openai-dev", "permission": [{"id": "modelperm-64LWHdlANgak2rHzc3K5Stt0", "object": "model_permission", "created": 1669085864, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "code-search-babbage-code-001", "parent": "None"}, {"id": "audio-transcribe-deprecated", "object": "model", "created": 1674776185, "owned_by": "openai-internal", "permission": [{"id": "modelperm-IPCtO1a9wW5TDxGCIqy0iVfK", "object": "model_permission", "created": 1674776185, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "audio-transcribe-deprecated", "parent": "None"}, {"id": "code-davinci-002", "object": "model", "created": 1649880485, "owned_by": "openai", "permission": [{"id": "modelperm-mBXN1A2dsyXVzzhzbkBNm2L3", "object": "model_permission", "created": 1674446967, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "code-davinci-002", "parent": "None"}, {"id": "text-ada-001", "object": "model", "created": 1649364042, "owned_by": "openai", "permission": [{"id": "modelperm-KN5dRBCEW4az6gwcGXkRkMwK", "object": "model_permission", "created": 1669088497, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-ada-001", "parent": "None"}, {"id": "text-similarity-ada-001", "object": "model", "created": 1651172505, "owned_by": "openai-dev", "permission": [{"id": "modelperm-DdCqkqmORpqxqdg4TkFRAgmw", "object": "model_permission", "created": 1669092759, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-similarity-ada-001", "parent": "None"}, {"id": "text-davinci-insert-002", "object": "model", "created": 1649880484, "owned_by": "openai", "permission": [{"id": "modelperm-V5YQoSyiapAf4km5wisXkNXh", "object": "model_permission", "created": 1669066354, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-davinci-insert-002", "parent": "None"}, {"id": "ada-code-search-code", "object": "model", "created": 1651172505, "owned_by": "openai-dev", "permission": [{"id": "modelperm-wa8tg4Pi9QQNaWdjMTM8dkkx", "object": "model_permission", "created": 1669087421, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "ada-code-search-code", "parent": "None"}, {"id": "text-davinci-002", "object": "model", "created": 1649880484, "owned_by": "openai", "permission": [{"id": "modelperm-68gqXLT2Fmu37alneH4toXtF", "object": "model_permission", "created": 1674172873, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-davinci-002", "parent": "None"}, {"id": "ada-similarity", "object": "model", "created": 1651172507, "owned_by": "openai-dev", "permission": [{"id": "modelperm-LtSIwCEReeDcvGTmM13gv6Fg", "object": "model_permission", "created": 1669092759, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "ada-similarity", "parent": "None"}, {"id": "code-search-ada-text-001", "object": "model", "created": 1651172507, "owned_by": "openai-dev", "permission": [{"id": "modelperm-JBssaJSmbgvJfTkX71y71k2J", "object": "model_permission", "created": 1669087421, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "code-search-ada-text-001", "parent": "None"}, {"id": "text-search-ada-query-001", "object": "model", "created": 1651172505, "owned_by": "openai-dev", "permission": [{"id": "modelperm-1YiiBMYC8it0mpQCBK7t8uSP", "object": "model_permission", "created": 1669092640, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-search-ada-query-001", "parent": "None"}, {"id": "text-curie-001", "object": "model", "created": 1649364043, "owned_by": "openai", "permission": [{"id": "modelperm-fGAoEKBH01KNZ3zz81Sro34Q", "object": "model_permission", "created": 1669066352, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-curie-001", "parent": "None"}, {"id": "text-davinci-edit-001", "object": "model", "created": 1649809179, "owned_by": "openai", "permission": [{"id": "modelperm-VzNMGrIRm3HxhEl64gkjZdEh", "object": "model_permission", "created": 1669066354, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-davinci-edit-001", "parent": "None"}, {"id": "davinci-search-document", "object": "model", "created": 1651172509, "owned_by": "openai-dev", "permission": [{"id": "modelperm-M43LVJQRGxz6ode34ctLrCaG", "object": "model_permission", "created": 1669066355, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "davinci-search-document", "parent": "None"}, {"id": "ada-code-search-text", "object": "model", "created": 1651172510, "owned_by": "openai-dev", "permission": [{"id": "modelperm-kFc17wOI4d1FjZEaCqnk4Frg", "object": "model_permission", "created": 1669087421, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "ada-code-search-text", "parent": "None"}, {"id": "text-search-ada-doc-001", "object": "model", "created": 1651172507, "owned_by": "openai-dev", "permission": [{"id": "modelperm-kbHvYouDlkD78ehcmMOGdKpK", "object": "model_permission", "created": 1669092640, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-search-ada-doc-001", "parent": "None"}, {"id": "code-davinci-edit-001", "object": "model", "created": 1649880484, "owned_by": "openai", "permission": [{"id": "modelperm-WwansDxcKNvZtKugNqJnsvfv", "object": "model_permission", "created": 1669066354, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "code-davinci-edit-001", "parent": "None"}, {"id": "davinci-instruct-beta", "object": "model", "created": 1649364042, "owned_by": "openai", "permission": [{"id": "modelperm-k9kuMYlfd9nvFiJV2ug0NWws", "object": "model_permission", "created": 1669066356, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "davinci-instruct-beta", "parent": "None"}, {"id": "text-babbage-001", "object": "model", "created": 1649364043, "owned_by": "openai", "permission": [{"id": "modelperm-hAf2iBGMqLmqB9HZiwrp1gL7", "object": "model_permission", "created": 1669086409, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-babbage-001", "parent": "None"}, {"id": "text-similarity-curie-001", "object": "model", "created": 1651172507, "owned_by": "openai-dev", "permission": [{"id": "modelperm-6dgTTyXrZE7d53Licw4hYkvd", "object": "model_permission", "created": 1669079883, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-similarity-curie-001", "parent": "None"}, {"id": "code-search-ada-code-001", "object": "model", "created": 1651172507, "owned_by": "openai-dev", "permission": [{"id": "modelperm-8soch45iiGvux5Fg1ORjdC4s", "object": "model_permission", "created": 1669087421, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "code-search-ada-code-001", "parent": "None"}, {"id": "ada-search-query", "object": "model", "created": 1651172505, "owned_by": "openai-dev", "permission": [{"id": "modelperm-b753xmIzAUkluQ1L20eDZLtQ", "object": "model_permission", "created": 1669092640, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "ada-search-query", "parent": "None"}, {"id": "text-search-davinci-query-001", "object": "model", "created": 1651172505, "owned_by": "openai-dev", "permission": [{"id": "modelperm-9McKbsEYSaDshU9M3bp6ejUb", "object": "model_permission", "created": 1669066353, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-search-davinci-query-001", "parent": "None"}, {"id": "curie-similarity", "object": "model", "created": 1651172510, "owned_by": "openai-dev", "permission": [{"id": "modelperm-z9GtwMD6HcxKqvsPfDm0PSg6", "object": "model_permission", "created": 1669079884, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "curie-similarity", "parent": "None"}, {"id": "davinci-search-query", "object": "model", "created": 1651172505, "owned_by": "openai-dev", "permission": [{"id": "modelperm-lYkiTZMmJMWm8jvkPx2duyHE", "object": "model_permission", "created": 1669066353, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "davinci-search-query", "parent": "None"}, {"id": "text-davinci-insert-001", "object": "model", "created": 1649880484, "owned_by": "openai", "permission": [{"id": "modelperm-3gRQMBOMoccZIURE3ZxboZWA", "object": "model_permission", "created": 1669066354, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-davinci-insert-001", "parent": "None"}, {"id": "babbage-search-document", "object": "model", "created": 1651172510, "owned_by": "openai-dev", "permission": [{"id": "modelperm-5qFV9kxCRGKIXpBEP75chmp7", "object": "model_permission", "created": 1669084981, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "babbage-search-document", "parent": "None"}, {"id": "ada-search-document", "object": "model", "created": 1651172507, "owned_by": "openai-dev", "permission": [{"id": "modelperm-8qUMuMAbo4EwedbGamV7e9hq", "object": "model_permission", "created": 1669092640, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "ada-search-document", "parent": "None"}, {"id": "text-davinci-003", "object": "model", "created": 1669599635, "owned_by": "openai-internal", "permission": [{"id": "modelperm-vb2l2yoJf2A78bxHfhSp5doO", "object": "model_permission", "created": 1674848581, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-davinci-003", "parent": "None"}, {"id": "curie", "object": "model", "created": 1649359874, "owned_by": "openai", "permission": [{"id": "modelperm-NvPNUvr0g9gAt3B6Uw4sZ2do", "object": "model_permission", "created": 1669080023, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "curie", "parent": "None"}, {"id": "text-search-babbage-doc-001", "object": "model", "created": 1651172509, "owned_by": "openai-dev", "permission": [{"id": "modelperm-ao2r26P2Th7nhRFleHwy2gn5", "object": "model_permission", "created": 1669084981, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-search-babbage-doc-001", "parent": "None"}, {"id": "text-search-curie-doc-001", "object": "model", "created": 1651172509, "owned_by": "openai-dev", "permission": [{"id": "modelperm-zjXVr8IzHdqV5Qtg5lgxS7Ci", "object": "model_permission", "created": 1669066353, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-search-curie-doc-001", "parent": "None"}, {"id": "text-search-curie-query-001", "object": "model", "created": 1651172509, "owned_by": "openai-dev", "permission": [{"id": "modelperm-a58jAWPMqgJQffbNus8is1EM", "object": "model_permission", "created": 1669066357, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-search-curie-query-001", "parent": "None"}, {"id": "babbage-search-query", "object": "model", "created": 1651172509, "owned_by": "openai-dev", "permission": [{"id": "modelperm-wSs1hMXDKsrcErlbN8HmzlLE", "object": "model_permission", "created": 1669084981, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "babbage-search-query", "parent": "None"}, {"id": "text-search-davinci-doc-001", "object": "model", "created": 1651172505, "owned_by": "openai-dev", "permission": [{"id": "modelperm-qhSf1j2MJMujcu3t7cHnF1DN", "object": "model_permission", "created": 1669066353, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-search-davinci-doc-001", "parent": "None"}, {"id": "text-search-babbage-query-001", "object": "model", "created": 1651172509, "owned_by": "openai-dev", "permission": [{"id": "modelperm-Kg70kkFxD93QQqsVe4Zw8vjc", "object": "model_permission", "created": 1669084981, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-search-babbage-query-001", "parent": "None"}, {"id": "curie-search-document", "object": "model", "created": 1651172508, "owned_by": "openai-dev", "permission": [{"id": "modelperm-1xwmXNDpvKlQj3erOEVKZVjO", "object": "model_permission", "created": 1669066353, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "curie-search-document", "parent": "None"}, {"id": "text-similarity-davinci-001", "object": "model", "created": 1651172505, "owned_by": "openai-dev", "permission": [{"id": "modelperm-OvmcfYoq5V9SF9xTYw1Oz6Ue", "object": "model_permission", "created": 1669066356, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-similarity-davinci-001", "parent": "None"}, {"id": "audio-transcribe-001", "object": "model", "created": 1656447449, "owned_by": "openai", "permission": [{"id": "modelperm-DEyvUa4t6g4mVL1AmmtB0SHO", "object": "model_permission", "created": 1669066355, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "audio-transcribe-001", "parent": "None"}, {"id": "davinci-similarity", "object": "model", "created": 1651172509, "owned_by": "openai-dev", "permission": [{"id": "modelperm-lYYgng3LM0Y97HvB5CDc8no2", "object": "model_permission", "created": 1669066353, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "True", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "davinci-similarity", "parent": "None"}, {"id": "cushman:2020-05-03", "object": "model", "created": 1590625110, "owned_by": "system", "permission": [{"id": "snapperm-FAup8P1KqclNlTsunLDRiesT", "object": "model_permission", "created": 1590625111, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "True", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "cushman:2020-05-03", "parent": "None"}, {"id": "ada:2020-05-03", "object": "model", "created": 1607631625, "owned_by": "system", "permission": [{"id": "snapperm-9TYofAqUs54vytKYL0IX91rX", "object": "model_permission", "created": 1607631626, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "ada:2020-05-03", "parent": "None"}, {"id": "babbage:2020-05-03", "object": "model", "created": 1607632611, "owned_by": "system", "permission": [{"id": "snapperm-jaLAcmyyNuaVmalCE1BGTGwf", "object": "model_permission", "created": 1607632613, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "babbage:2020-05-03", "parent": "None"}, {"id": "curie:2020-05-03", "object": "model", "created": 1607632725, "owned_by": "system", "permission": [{"id": "snapperm-bt6R8PWbB2SwK5evFo0ZxSs4", "object": "model_permission", "created": 1607632727, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "curie:2020-05-03", "parent": "None"}, {"id": "davinci:2020-05-03", "object": "model", "created": 1607640163, "owned_by": "system", "permission": [{"id": "snapperm-99cbfQTYDVeLkTYndX3UMpSr", "object": "model_permission", "created": 1607640164, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "davinci:2020-05-03", "parent": "None"}, {"id": "if-davinci-v2", "object": "model", "created": 1610745990, "owned_by": "openai", "permission": [{"id": "snapperm-58q0TdK2K4kMgL3MoHvGWMlH", "object": "model_permission", "created": 1610746036, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "if-davinci-v2", "parent": "None"}, {"id": "if-curie-v2", "object": "model", "created": 1610745968, "owned_by": "openai", "permission": [{"id": "snapperm-fwAseHVq6NGe6Ple6tKfzRSK", "object": "model_permission", "created": 1610746043, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "if-curie-v2", "parent": "None"}, {"id": "if-davinci:3.0.0", "object": "model", "created": 1629420755, "owned_by": "openai", "permission": [{"id": "snapperm-T53lssiyMWwiuJwhyO9ic53z", "object": "model_permission", "created": 1629421809, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "True", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "if-davinci:3.0.0", "parent": "None"}, {"id": "davinci-if:3.0.0", "object": "model", "created": 1629498070, "owned_by": "openai", "permission": [{"id": "snapperm-s6ZIAVMwlZwrLGGClTXqSK3Q", "object": "model_permission", "created": 1629498084, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "True", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "davinci-if:3.0.0", "parent": "None"}, {"id": "davinci-instruct-beta:2.0.0", "object": "model", "created": 1629501914, "owned_by": "openai", "permission": [{"id": "snapperm-c70U4TBfiOD839xptP5pJzyc", "object": "model_permission", "created": 1629501939, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "True", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "davinci-instruct-beta:2.0.0", "parent": "None"}, {"id": "text-ada:001", "object": "model", "created": 1641949608, "owned_by": "system", "permission": [{"id": "snapperm-d2PSnwFG1Yn9of6PvrrhkBcU", "object": "model_permission", "created": 1641949610, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-ada:001", "parent": "None"}, {"id": "text-davinci:001", "object": "model", "created": 1641943966, "owned_by": "system", "permission": [{"id": "snapperm-Fj1O3zkKXOQy6AkcfQXRKcWA", "object": "model_permission", "created": 1641944340, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-davinci:001", "parent": "None"}, {"id": "text-curie:001", "object": "model", "created": 1641955047, "owned_by": "system", "permission": [{"id": "snapperm-BI9TAT6SCj43JRsUb9CYadsz", "object": "model_permission", "created": 1641955123, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-curie:001", "parent": "None"}, {"id": "text-babbage:001", "object": "model", "created": 1642018370, "owned_by": "openai", "permission": [{"id": "snapperm-7oP3WFr9x7qf5xb3eZrVABAH", "object": "model_permission", "created": 1642018480, "allow_create_engine": "False", "allow_sampling": "True", "allow_logprobs": "True", "allow_search_indices": "False", "allow_view": "True", "allow_fine_tuning": "False", "organization": "*", "group": "None", "is_blocking": "False"}], "root": "text-babbage:001", "parent": "None"}]}
engines = {"object": "list", "data": [{"object": "engine", "id": "babbage", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "ada", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "davinci", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-Embedding-ada-002", "ready": "True", "owner": "openai-internal", "permissions": "None", "created": "None"}, {"object": "engine", "id": "babbage-code-search-code", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-similarity-babbage-001", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-davinci-001", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "curie-instruct-beta", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "babbage-code-search-text", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "babbage-similarity", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "curie-search-query", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "code-search-babbage-text-001", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "code-cushman-001", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "code-search-babbage-code-001", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "audio-transcribe-deprecated", "ready": "True", "owner": "openai-internal", "permissions": "None", "created": "None"}, {"object": "engine", "id": "code-davinci-002", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-ada-001", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-similarity-ada-001", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-davinci-insert-002", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "ada-code-search-code", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-davinci-002", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "ada-similarity", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "code-search-ada-text-001", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-search-ada-query-001", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-curie-001", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-davinci-edit-001", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "davinci-search-document", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "ada-code-search-text", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-search-ada-doc-001", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "code-davinci-edit-001", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "davinci-instruct-beta", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-babbage-001", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-similarity-curie-001", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "code-search-ada-code-001", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "ada-search-query", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-search-davinci-query-001", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "curie-similarity", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "davinci-search-query", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-davinci-insert-001", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "babbage-search-document", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "ada-search-document", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-davinci-003", "ready": "True", "owner": "openai-internal", "permissions": "None", "created": "None"}, {"object": "engine", "id": "curie", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-search-babbage-doc-001", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-search-curie-doc-001", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-search-curie-query-001", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "babbage-search-query", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-search-davinci-doc-001", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-search-babbage-query-001", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "curie-search-document", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "text-similarity-davinci-001", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}, {"object": "engine", "id": "audio-transcribe-001", "ready": "True", "owner": "openai", "permissions": "None", "created": "None"}, {"object": "engine", "id": "davinci-similarity", "ready": "True", "owner": "openai-dev", "permissions": "None", "created": "None"}]}
cats = ['categories','parameters','specifications','descriptions','endpoints','paramNeeds','models','engines'],str('categories,parameters,specifications,descriptions,endpoints,paramNeeds,models,engines').split(',')
info ={"completions":{"categories":["chat","translate","qanda","parse"],
                      "choices":{"model":{'default': 'text-davinci-003', 'list': ['text-ada-001', 'text-davinci-003', 'text-curie-001', 'text-babbage-001']}},"specifications":{'chat':{'type': 'completions','refference':['completions','create'],"parameters":{'required':['model','prompt'],'optional':['user','stream','n','suffix','max_tokens','logit_bias','temperature','best_of','top_p','frequency_penalty','presence_penalty','stop','echo'],'prompt':{"task":"chat","structure": '','vars': {'prompt':{"input":"what would you like to say to the bot?","delimiter":''}}}}}}},
       "coding":{"categories":["editcode","debugcode","convertcode","writecode"],'choices':{'language':{'default':'python','list':['Python','Java','C++','JavaScript','Go','Julia','R','MATLAB','Swift','Prolog','Lisp','Haskell','Erlang','Scala','Clojure','F#','OCaml','Kotlin','Dart']},'models':{'default':'code-davinci-002','list':['code-cushman-001','text-davinci-003','code-davinci-002']}},"specifications":{'writecode':{'type': 'coding','refference':['completions','create'],"parameters":{'required':['model','prompt'],'optional':['user','stream','n','suffix','max_tokens','logit_bias','temperature','best_of','top_p','frequency_penalty','presence_penalty','stop','echo'],'prompt':{"task":"write code in [language] based off of specific [instruction]:","structure":"[prompt]-describe the code; [language] - specify the target language",'vars': {'instruction':{"input":"describe what you are looking for, be specific","delimiter":"instructuions:\n"},"language":{"input":"which language would you like the code to be written in?","delimiter":"language:\n,"}}}}},'editcode':{'type': 'coding','refference':['completions','create'],"parameters":{'required':['model','prompt'],'optional':['user','stream','n','suffix','max_tokens','logit_bias','temperature','best_of','top_p','frequency_penalty','presence_penalty','stop','echo'],'prompt':{"task":"edit code","structure":"edit [code] based off of specific [instructions]","vars":{"instruction":{"input":"provide specific instructions on what you are looking to have edited about this code:","delimiter":"instructions:\n"},"code":{"input":"enter the code you would like to have edited:","delimiter":"code:\n"}}}}},'debugcode':{'type': 'coding','refference':['completions','create'],"parameters":{'required':['model','prompt'],'optional':['user','stream','n','suffix','max_tokens','logit_bias','temperature','best_of','top_p','frequency_penalty','presence_penalty','stop','echo'],'prompt':{"task":"debug the code:","structure":"debug the following code:\n","vars":{"code":{"input":"the code you would like to have debugged","delimiter":""}}}}},'convertcode':{'type': 'coding','refference':['completions','create'],"parameters":{'required':['model','prompt'],'optional':['user','stream','n','suffix','max_tokens','logit_bias','temperature','best_of','top_p','frequency_penalty','presence_penalty','stop','echo'],'prompt':{"task":"convert code to another language:","structure":"convert the following [code] to [language]","vars":{"language":{"input":"the language you would like the code converted to:","delimiter":"language:\n"},"code":{"input":"the code you would like to have converted","delimiter":"code:\n"}}}}}}},
       "images":{"categories":["image_create","image_edit","image_variatoin"],'choices':{"response_format":{"default":"url","list":['url','b64_json']},"size":{"default":"1024x1024","list":["256x256","512x512","1024x1024"]}},"specifications":{'image_variation':{'type': 'images',"refference":["image","create","image_variation"],"parameters":{"required":["image"],'optional':["prompt",'size','n','response_format'],'prompt':{"task":"image variation","structure":"create a variation of the [image] based off of [instructions] if given:\n","vars":{"instruction":{"input":"describe what you would like to have done with the image(s):","delimiter":"instructions:\n"}}}}},'image_create':{'type': 'images',"refference":["image","create","image_create"],"parameters":{'required':['prompt'],'optional':['size','n','response_format'],'prompt':{"task":"image creation","structure":"create an image based on the following [instructions]:\n","vars":{"instructions":{"input":"describe the image you would like to create:","delimiter":"instructions:\n"}}}}},"image_edit":{'type':'images',"refference":["image","create","image_edit"],"parameters":{'required':['image','prompt'],'optional':['mask','size','n','response_format'],"prompt":{"task":"image creation","structure":'[image]-main image; [mask] secondary image;[prompt]- input how you would like to have it edited',"vars":{"instructions":{"input":"provide instructions describing what you would like to have done with the image(s):","delimiter":"instructions:"}}}}}}},


       "edit":{"categories":["edit"],"choices":{"model":{'default': 'text-ada-001', 'list': ['text-ada-001', 'text-davinci-003', 'text-curie-001', 'text-babbage-001']}},"specifications":{'edits':{'type': 'edits','refference':['edits','create'],"parameters":{'required':['model','prompt'],'optional':['user','stream','n','suffix','max_tokens','logit_bias','temperature','best_of','top_p','frequency_penalty','presence_penalty','stop','echo'],'prompt':{"task":"edit text","structure":'[image]-main image; [mask] secondary image;[prompt]- input how you would like to have it edited',"vars":{"instructions":{"input":"provide instructions describing what you would like to have edited:","delimiter":"instructions:"}}}}}}},"moderation":{"categories":["moderate"],'choices':{'models':{'default':"text-davinci-003",'list':["text-moderation-001","text-davinci-003"]}},"specifications":{"moderate":{'type': 'moderation',"refference":["completions","moderation"],"parameters":{'required':['input'],'optional':['model'],'prompt':{"task":"moderation","structure":'text to moderate:\n',"vars":{"text":{"input":"provide the text you would like to have moderated","delimiter":""}}}}}}}}



changeGlob('specializedSet',{'inputTabKeys':{'types':[],'inputLs':[],'index':1,'names':[],'descriptions':[]},'jsList':["g.Text('Parameters', font='Any 20')],[txtBox('def  '),txtBox('dis  '),txtBox('   inf  ')]"],'prevKeys':[],'userMgs':'','resp':'','content':'application/json'})

#specializedSet = {'inputTabKeys':{'types':[],'inputLs':[],'index':1,'names':[],'descriptions':[]},'jsList':[[sg.Text('Parameters', font='Any 20')],[txtBox('def  '),txtBox('dis  '),txtBox('   inf  ')]],'prevKeys':[],'userMgs':'','resp':'','content':'application/json'}
changeGlob('categoriesJs',getAllInfo('categories'))
changeGlob('catKeys',fun.getKeys(categoriesJs))
changeGlob('category',catKeys[0])
changeGlob('specializedLs',categoriesJs[category])
changeGlob('specialization',specializedLs[0])
changeGlob('tabIndex',[])
getAllThings()
#chatInput = [[sg.Text('chatInput', font='Any 20')],txtInput('chat Input',100,10),[sg.Button('SEND', button_color=(sg.YELLOWS[0], sg.BLUES[0]), bind_return_key=True),sg.Button('EXIT', button_color=(sg.YELLOWS[0], sg.GREENS[0]))]]



