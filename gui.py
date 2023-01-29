import PySimpleGUI as sg
def fileBrowser():
  sg.theme("DarkTeal2")
  layout = [[sg.T("")], [sg.Text("Choose a file: "), sg.Input(key="-IN2-" ,change_submits=True), sg.FileBrowse(key="-IN-")],[sg.Button("Submit")]]
  window = sg.Window('My File Browser', layout, size=(600,150))   
  while True:
      event, values = window.read()
      if event == sg.WIN_CLOSED or event=="Exit":
          break
      elif event == "Submit":
        break
        return values["-IN-"]
        
