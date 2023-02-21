
import functions as fun
import guiFunctions as guiFun
import PySimpleGUI as sg
import json
from datetime import date
import guiFunctions as guiFun
import os
import openai
import demoImgh64
import requests
import os.path
from os import path
def changeGlob(x,y):
    globals()[x]=y
    return y
changeGlob('jsFrames',{'top_banner':{'pad': (0,0), 'background_color': '#1B2838', 'expand_x': True, 'border_width': 0, 'grab': True},
                       'top':{'pad': ((20,20), (20, 10)), 'expand_x': True, 'relief': sg.RELIEF_GROOVE, 'border_width': 3},
                       'previousQuery':{'pad': (0, (10, 0)), 'border_width': 0, 'expand_x': True, 'expand_y': True},
                       'querySection':{'pad': (0, (10, 0)), 'border_width': 0, 'expand_x': True, 'expand_y': True},
                       'chatInput':{"title":"Chat Input",'pad': (0, (10, 0)), 'border_width': 0, 'expand_x': True, 'expand_y': True},
                       'outputSection':{"title":"Chat Ou",'pad': (0, (10, 0)), 'border_width': 0, 'expand_x': True, 'expand_y': True},
                       'parameterSection':{'pad': ((10,20), (10, 0)), 'border_width': 0, 'expand_x': True, 'expand_y': True}})
  
def mkStraight(ls):
  lsN = []
  for k in range(0,len(ls)):
    if fun.isLs(ls[k]):
      for i in range(0,len(ls[k])):
        lsN.append(ls[k][i])
    else:
      lsN.append(ls[k])
  return mkAllDiv(lsN)
def mkAllDiv(ls):
  return [[[ls]]]
def getInput():
    print('getInput')
    
    txt0 = guiFun.txtInputs({"title":'Chat Input',"size":(50,10),"font":'Any 20',"key":'structure',"autoscroll":True,"disabled":False})
    vertSep = guiFun.vertSep({"pad":None})
    butt0 = guiFun.getButton({"button_text":'Compile',"visible":True,"key":'Compile',"enable_events":True," button_color":templateJs['BUTT_COLOR_Y_B'],"bind_return_key":True})
    butt1 = guiFun.getButton({"button_text":'SEND',"visible":True,"key":'SEND',"enable_events":True," button_color":templateJs['BUTT_COLOR_Y_B'],"bind_return_key":True})
    butt2 = guiFun.getButton({"button_text":'EXIT',"visible":True,"key":'EXIT',"enable_events":True," button_color":templateJs['BUTT_COLOR_Y_G'],"bind_return_key":True})
    struct = mkStraight([mkStraight([txt0,vertSep]),mkStraight([butt0,butt1,butt2])])#,[[butt0],[butt1],[butt2]]]
    return struct
global category,specializations,categories,specialization,catKeys,specializedSet,catLs,categoriesJs,specializations,info,tabIndex,parameters, pastedinwindow
import infoSheets as infoS
from infoSheets import mid,categories,parameters,specifications,choi,descriptions,endpoints,paramNeeds,models,engines,cats,info
changeGlob('templateJs',{'BPAD_BANNER':(0,0),'BORDER_COLOR':'#C7D5E0','DARK_HEADER_COLOR':'#1B2838','BPAD_TOP':((20,20), (20, 10)),'BPAD_LEFT':((20,10), (0, 0)),'BPAD_LEFT_INSIDE':(0, (10, 0)),'BPAD_RIGHT':((10,20), (10, 0)),'BUTT_COLOR_Y_G':(sg.YELLOWS[0], sg.GREENS[0]),'BUTT_COLOR_Y_B':(sg.YELLOWS[0], sg.BLUES[0])})

#[top_banner],[[top],[sg.Frame('', [[quFr],[prQu],[outputSection],[chatInput]],pad=templateJs['BPAD_LEFT'], background_color=templateJs['BORDER_COLOR'], border_width=0, expand_x=True, expand_y=True),params(),],[sg.Sizegrip(background_color=templateJs['BORDER_COLOR'])]]]
window = sg.Window('Dashboard PySimpleGUI-Style', getInput())
while True:
          event, values = window.read()

