# openAI-chat-GPT-bot
python tools for chat gpt3
#README

This code is a collection of functions that are used to create a prompt for user input. The code includes functions for requesting input, creating prompts, finding specific items within a list, and appending items to a list. 

The getCall function is used to request a specific type of response from the user. This includes prompting the user to select a file, input a value, or select a model. The ifNoThenApp function is used to append an item to a list if it is not already present. The getAllNew function is used to loop through a list and append any items that are not already present. 

The createAsk and findIt functions are used to create prompts for the user to select from and find specific items within a list, respectively. The askIt function is used to prompt the user to select from a list of choices. 

Overall, this code is useful for prompting the user for input and providing a way for them to select from a list of options.

This code is written in Python and is used to call an API from a given endpoint. It uses the PIL library to work with image files and pars to work with endpoints, models, engines, files, parameters, variables, parameter needs, prompts, ends, and specifications. The code contains a function askBoth that returns a list of two strings. It then uses the callJson function to make the API call. 

The code begins by importing the PIL library and setting the content type to “application/json”. It then imports the pars library and changes the content type to “multipart/form-data”. Finally, it uses the askBoth function to get a list of two strings and then calls the callJson function with the list as a parameter. 

The askBoth function uses the askIt function to get a string from the user. It first gets the string from a list of “ends” and then gets a second string from a list of endpoints for the “ends”. Finally, it gets a third string from a list of specifications for the “ends”. It then returns a list containing the three strings. 

The askIt function takes in two parameters: a list of strings and a prompt string. It prints out the prompt string and then a list of the strings in the list. The user is then prompted to enter an integer that corresponds to the string in the list they would like to choose. If the user enters an invalid input, -1 is returned. Otherwise, the corresponding string is returned. 

The callJson function takes a list of two strings as a parameter and then makes an API call using the two strings. It returns the response from the API call.

This is a collection of Python functions designed to help with a variety of tasks. 

The functions include: 

- `changeGlob`: takes two arguments and changes the value of a global variable 
- `getFile`: prompts the user to select a file 
- `mkDir`: creates a directory 
- `crPa`: creates a path 
- `isDir`: checks if a directory exists 
- `isFile`: checks if a file exists 
- `pen`: writes to a file 
- `reader`: reads from a file 
- `jsRead`: reads from a file and converts it to a JSON object 
- `jsItRead`: reads from a file, converts it to a string, and then converts it to a JSON object 
- `jsIt`: takes a JSON string and converts it to a JSON object 
- `jsItSpec`: takes a special JSON string and converts it to a JSON object 
- `makeQuote`: adds quotation marks to a string 
- `whileIn`: takes a string and removes all instances of a character from the string 
- `numLs`: returns a list of numbers 
- `isNum`: checks if a string is a number 
- `isLs`: checks if an object is a list 
- `isStr`: checks if an object is a string 
- `isInt`: checks if an object is an integer 
- `isFloat`: checks if an object is a float 
- `isBool`: checks if an object is a boolean 
- `mkF`: creates a file 
- `cv2e`: reads an image file using OpenCV 
- `cv2ec`: converts an image file from BGR to RGB using OpenCV 
- `cv2es`: saves an image file using OpenCV 

These functions are designed to make it easier to work with files, create paths, and convert data.

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

This code provides a set of functions for dealing with different types of data. The mkFloat() function takes a number as input and returns a float. The mkBool() function takes a boolean as input and returns a boolean. The mkStr() function takes a string as input and returns a string. The getObjObj() function takes an object and a value as input and returns an output based on the type of object. The getKeys() function takes a JavaScript object as input and returns a list of keys. The isLs() function takes a list as input and returns a list. The getAllDefs() function takes no input and returns a set of parameters. The getJson() and getText() functions take a list as input and return a JSON or text object, respectively. The callJson() and callText() functions take a list as input and return a JSON or text object, respectively. The uploadFile() function takes a file and a purpose as input and uploads the 

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
fault = '+str(default)+'\ntype: '+str(type)+'\nobject: '+str(object)+f'\nscale: {ran}:\nplease input your response for '+str(key)+'(leave blank for default):\n'


This code is a set of functions that interact with the OpenAI API in order to retrieve and manipulate content. The functions include create() which creates a file and stores it in OpenAI, retrieveFile() which retrieves a file from OpenAI, retrieveContent() which downloads a file from OpenAI, listFiles() which lists all files in OpenAI, GETHeader() which creates the HTTP headers for GET requests, reqGet() which performs a GET request, reqPost() which performs a POST request, reqDelete() which performs a DELETE request, reqImageVariation() which creates a variation of an image, reqImageEdit() which edits an image, retrieveModel() which retrieves a model from OpenAI, getModels() which retrieves a list of models from OpenAI, getEngines() which retrieves a list of engines from OpenAI, getFiles() which retrieves a list of files from OpenAI, getSends() which retrieves a specific piece of data from a dictionary, correctJson() which corrects a json object, getAllDefaults() which retrieves all default parameters and boolAsk() which asks a boolean question. All of these functions interact with the OpenAI API to create and manipulate content.,obj)
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

This code is part of a larger program that is used to generate prompts for users. The code includes two functions, getContext and getPrompt, that are used to get user input. 

The getContext function takes in several parameters, such as a scale, a key, a default, a type, and an object. It then generates a prompt string which includes the key, options, default, type and object. It also includes range information depending on the scale. This prompt string is then returned.

The getPrompt function takes in a scale and a JavaScript object. It then generates a prompt string by looping through the keys of the JavaScript object and using the getContext function to generate the prompt. It also checks for a cycle to determine if the user needs to input more than one value. Finally, the generated prompt string is returned. 

The getPromptVars function takes in a list of parameters and uses the getSends and boolAsk functions to generate a prompt string. It takes the parameters from the list and uses the getContext function to generate a prompt string. It then checks if the user only wants to include required parameters and uses the input function to get the user's response. Finally, the response is stored in the list and the list is returned.
