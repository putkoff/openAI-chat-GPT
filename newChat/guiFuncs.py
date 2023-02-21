import PySimpleGUI as sg
import functions as fun
import tkinter as tk
import tkinter.ttk as ttk
import functions as fun
from tktooltip import ToolTip
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
def changeGlob(x,y):
    globals()[x] = y
    return y
def ifThenBestOf(na,ev):
    if ev == 'evEnSl':
         if na in ['best_of','n']:
            return True
         return True
def ifBig(x):
    if fun.isInt(x) and x<=10 :
        return -int(1)
    else:
        return float(x)/5
def updateValues(wi,x,y):
    wi.find_element(x).update(value=y)
def getDropLs(na,ls):
    return [sg.Combo(ls),sg.Text(na)]
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
def imgScreen(x):
    pil_image = Image.open(io.BytesIO(x))
    png_bio = io.BytesIO()
    pil_image.save(png_bio, format="PNG")
    png_data = png_bio.getvalue()
    return [[sg.Text(size=(40, 1), key="-TOUT-")],[sg.Image(data=png_data, key="-ArtistAvatarIMG-")]]
def tabGroup(na,ls):
    return sg.Tab(na, ls, tooltip='tip')
def scriptOutput():
    return [[sg.Text('Script output....', size=(40, 1))],[sg.Output(size=(88, 20), font='Courier 10')]]
def getMenu(ls):
    return [sg.Menu(ls)]
def getChkBox(na,defa):
    return [sg.Checkbox(na, size=(12, 1), default=defa,key=na)]
def getLsSect(js):
    return [sg.vtop(sg.Listbox(list(range(10)), size=(5,5),key=na,default_value="None",)), sg.Multiline(size=(25,10)),sg.Text(na)]#,[sg.InputOptionMenu(('Menu Option 1', 'Menu Option 2', 'Menu Option 3'))]
def getBrwsSect(na,desc,location):
    return [[sg.T(desc)],[sg.Text(na), sg.Input(change_submits=True,key=na), sg.FileBrowse(file_types=(("Image Files", "*.png"),),key=na)],[sg.Button("Submit")]]
def getTxtBox(na,desc):
    return  [[sg.Text(desc)],[sg.Multiline(size=(50,10), font='Tahoma 13', key=na, autoscroll=True),sg.VerticalSeparator(pad=None)]]
def getFile():
    return [sg.Text('Choose A Folderg', size=(35, 1))]
def getSlider(na,ls,typ,defa):
    if typ == 'float':
        return [[sg.Slider(range=(float(ls[0]),float(ls[1])),  key=na,default_value=float(defa),resolution=-100, tick_interval=1,pad= None,orientation='h',disable_number_display=False, enable_events=False),sg.Checkbox('default', enable_events=True, key=na+'_ch', size=(4, 1)),sg.Button(na, border_width=4, key=na+'_Info',tooltip="get info"),]]
    return [sg.Slider(range=(int(ls[0]),int(ls[1])),key=na, orientation='h', default_value=int(defa),resolution=-ifBig(int(ls[1])), tick_interval=ifBig(int(ls[1])), enable_events=ifThenBestOf(na,'evEnSl')),sg.Checkbox('default', size=(4, 1)),sg.Button(na, enable_events=True,border_width=4, key=na+'_Info',tooltip="get info")]
def getMenuLs(na,ls,defa):
    return [sg.Text(na),sg.Combo(ls,key=na,default_value=defa),sg.Push()]
def getPopup(text):
    return [sg.Text(text,key=na)]
def getInput(na,st):
    return [sg.Text(na),sg.InputText(st,key=na),sg.Push()]
def getButtons(butts):
  butts,lsN = fun.mkLs(butts),[]
  for i in range(0,len(butts)):
    lsN.append(sg.Button(butts[i]))
  return lsN
def getKeys(js):
  lsN = []
  try:
    for key in js.keys():
      lsN.append(key)
    return lsN
  except:
    return lsN
def getVals(js):
  lsN = []
  try:
    for key in js.values():
      lsN.append(key)
    return lsN
  except:
    return lsN
def mkTxtLs(beg,ls):
  for i in range(0,len(ls)):
    beg = beg +str(ls[i])+'\n'
  return beg
def mkInputs(ls,prev,title,desc):
  lsA=[sg.Text(desc)]
  for i in range(0,len(ls)):
    lsA.append(getInput(ls[i],prev[i]))
  return defaultWindow(lsA,title,title2)
def getDrop(js,st):
  if fun.isLs(st):
      changeGlob('varDesc',st[0])
      st= st[1]
  keys,lsA = getKeys(js),[]
  for i in range(0,len(keys)):
    key = keys[i]
    lsA.append(getMenuLs(key,js[key],js[key][0]))
  return defaultWindow(lsA,st,st)
def checkValues(values):
  keys,error,er,errKeys = getKeys(values)[1:],'you have not selected an input for the following:\n',True,[]
  for i in range(0,len(keys)):
    key = keys[i]
    if values[key] == '':
      errKeys.append(key)
      er = False
  if er == False:
    popUp('error: values unchecked',mkTxtLs(error,errKeys))
    return False
  return True
def createListFromJs(js,keys,var):
    beg = 'descriptions of selected values below:\n'
    for i in range(0,len(keys)):
        if keys[i] in js:
            beg = beg + str(keys[i])+ ' - '+str(js[keys[i]])+'\n'
    return beg
def eventCall(window):
  event,values = window.read()
  if 'defVals' =={}:
      values['defVals'] = values
  values['currVals'] = values
  curr = values['currVals']
  if event == sg.WIN_CLOSED or event == 'Exit':
    window.close()
    return 'exit'
  elif '_Info' in str(event):
      from dataSheets import parameters
      popUp(event[:-len('_Info')],parameters[event[:-len('_Info')]]['description'])
  elif '_ch' in str(event):
      if values[event] == True:
          updateValues(window,event[:-len('_ev')],values['defVals'][:-len('_ev')])
  elif event in ['best_of','n']:
      if 'best_of' in fun.getKeys(values):
          if curr['n']>=curr['best_of']:
                  updateValues(window,'n',values['best_of']-1)
  elif event == 'override':
      window.close()
      return values
  elif event == '_Info':
      if varDesc != None:
          from dataSheets import dataSheets
          popUp('variable descriptions',createListFromJs(defVals,getKeys(values),'description'))
  elif event == 'OK':
    if checkValues(values):
      window.close()
      return values
  elif event == 'Run':
    if checkValues(values):
        changeGlob('gogo',False)
        return values
  if 'defVals' not in values:
      values['defVals'] = {}    
  return False
def popUp(title,text):
  window = sg.popup(title,text)
def defaultOverWindow(sg1,title):
    getDefaultSetOptions()
    window = sg.Window(title, getDefaultLayout(sg1), finalize=False)
    while True:
      vals = eventCall(window)
      if vals != False:
        return vals
def defaultWindow2(sg1,title):
    gogo = True
    getDefaultSetOptions()
    layout = [getDefaultLayout(sg1)]
    window = sg.Window(title,layout , finalize=False)
    while gogo == True:
      vals = eventCall(window)
      if vals != False:
        return vals
def defaultWindow(sg1,title1,title2):
    getDefaultSetOptions()
    layout = [[sg.Text(title2), sg.Text('', key='-OUTPUT-')],[sg.T('0',size=(4,1), key='-LEFT-'),sg1,sg.T('0', size=(4,1), key='-RIGHT-')],[sg.Button('OK'),sg.Button('Run'),sg.Button('Info'),sg.Button('Show'), sg.Button('Exit')]]
    window = sg.Window(title1, layout)
    while True:
      vals = eventCall(window)
      if vals != False:
        return vals
def formatted(texts,sliders,title):
    window = sg.Window('Columns')                                   # blank window
    col = [[sg.Text('col Row 1')],texts]
    layout = [[sliders, sg.Column(col)],[sg.In('Last input')],[sg.OK()]]
    window = sg.Window('Compact 1-line window with column', layout)
    event, values = window.read()
    window.close()
    sg.Popup(event, values, line_width=200)

