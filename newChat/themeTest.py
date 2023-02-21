import functions as fun
import PySimpleGUI as sg
import NewGuiButtons as nButt
def defaultWindow(sg1,title1,title2):
    #getDefaultSetOptions()
    #layout = [[sg.Text(title2), sg.Text('', key='-OUTPUT-')],[sg.T('0',size=(4,1), key='-LEFT-'),sg1,sg.T('0', size=(4,1), key='-RIGHT-')],[sg.Button('OK'),sg.Button('Run'),sg.Button('Info'),sg.Button('Show'), sg.Button('Exit')]]
    window = sg.Window('',sg1)
    while True:
      event,values = window.read()
      if vals != False:
        return vals
defaultWindow(Titlebar({}),'hey','ho')
