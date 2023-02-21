import PySimpleGUI as sg
import webbrowser
import urllib.parse

"""
    Beta of the GitHub Issue Post Code
    
    This program is going to be inside of PySimpleGUI itself.
    
    It produces Markdown code that 
    
    
"""

def _github_issue_post_make_markdown(issue_type, operating_system, os_ver, psg_port, psg_ver, gui_ver, python_ver,
                                     python_exp, prog_exp, used_gui, gui_notes,
                                     cb_docs, cb_demos, cb_demo_port, cb_readme_other, cb_command_line, cb_issues, cb_github,
                                     detailed_desc, code, ):
    body = \
"""
### Type of Issue (Enhancement, Error, Bug, Question)
{}
----------------------------------------
#### Operating System
{}  version {}
#### PySimpleGUI Port (tkinter, Qt, Wx, Web)
{}
----------------------------------------
## Versions
Version information can be obtained by calling `sg.main_get_debug_data()`
Or you can print each version shown in ()
#### Python version (`sg.sys.version`)
{}
#### PySimpleGUI Version (`sg.__version__`)
{}
#### GUI Version  (tkinter (`sg.tclversion_detailed`), PySide2, WxPython, Remi)
{}
""".format(issue_type, operating_system,os_ver, psg_port,python_ver, psg_ver, gui_ver)

    body2 = \
"""
---------------------
#### Your Experience In Months or Years (optional)
{} Years Python programming experience
{} Years Programming experience overall
{} Have used another Python GUI Framework? (tkinter, Qt, etc) (yes/no is fine)
{}
---------------------
#### Troubleshooting
These items may solve your problem. Please check those you've done by changing - [ ] to - [X]
- [{}] Searched main docs for your problem  www.PySimpleGUI.org
- [{}] Looked for Demo Programs that are similar to your goal Demos.PySimpleGUI.org
- [{}] If not tkinter - looked for Demo Programs for specific port
- [{}] For non tkinter - Looked at readme for your specific port if not PySimpleGUI (Qt, WX, Remi)
- [{}] Run your program outside of your debugger (from a command line)
- [{}] Searched through Issues (open and closed) to see if already reported Issues.PySimpleGUI.org
- [{}] Tried using the PySimpleGUI.py file on GitHub. Your problem may have already been fixed but not released
#### Detailed Description
{}
#### Code To Duplicate
A **short** program that isolates and demonstrates the problem (Do not paste your massive program, but instead 10-20 lines that clearly show the problem)
This pre-formatted code block is all set for you to paste in your bit of code:
```python
{}
```
#### Screenshot, Sketch, or Drawing
    """.format(python_exp, prog_exp, used_gui, gui_notes,
                cb_docs, cb_demos, cb_demo_port, cb_readme_other, cb_command_line, cb_issues, cb_github,
                detailed_desc, code if len(code) > 10 else '# Paste your code here')

    return body + body2





def _github_issue_post_make_github_link(title, body):
    pysimplegui_url = "https://github.com/PySimpleGUI/PySimpleGUI"
    pysimplegui_issues = f"{pysimplegui_url}/issues/new?"

    # Fix body cuz urllib can't do it smfh
    getVars = {'title': str(title), 'body': str(body)}
    return (pysimplegui_issues + urllib.parse.urlencode(getVars).replace("%5Cn", "%0D"))


#########################################################################################################
def getFullParamDesc(na):
    req= 'optional'
    if na in getParamNeeds(category,specialization)['required']:
        req = 'required'
    return 'name:'+str(na)+',  | default:'+str(req)+' | type:'+str(getObj(na))+'\ndescription: '+str(getParamDesc(na))
def getParamDesc(na):
    returnParameters(category,specialization)
    return parameters[na]['description']
def getObj(x):
    return parameters[x]['object']
def getDef(na):
    return mkType([parameters[na]['default'],parameters[na]['object']])
def getCurrVal(values,na):
    return mkType([values[na],parameters[na]['object']])
def getPrompt(cat,spec):
    return info[cat]["specifications"][spec]["parameters"]['prompt']
def getPromptVars(cat,spec):
    return getKeys(getPrompt(cat,spec))
def isParam(na):
    if na in parameters:
        return True
def getDefa(na):
    return getDef(na),na+'__default__'+str(getDef(na))
def ifOnTurnOff(event):
    if values[event]:
        return False
    return True
def boolDefa(values,na):
    if getCurrVal(values,na) == getDef(na):
        return True
    return False
def updateCat(cat):
    changeGlob('category',cat)
    changeGlob('specializedLs',categoriesJs[category])
    changeGlob('specialization',specializedLs[0])
    return getPromptVars(category,specialization)
def updateGuiDispVars():
    refWin(window)
    js,catLs = getAllThings(),categoriesJs[category]
    window['categoryDisplay'].update(value=js['category']['names'])
    window['categoryDescription'].update(value=js['categoryDefinition'])
    window["catCombo"].update(values = catLs)
    window["catCombo"].update(value=catLs[0])
    window['structure'].update(value=js['structure']['parse'])
    window['specializationDisplay'].update(value=js['specialization']['names'])
    window['specializationDescription'].update(value=js['specializationDeffinition'])
    refWin(window)
def refWin(window):
    window.refresh()
def visibility(boolIt,na,window):
    window[na].update(visible=boolIt)
def winDisable(boolIt,na,window):
    window[na].update(disabled=boolIt)
def winGet(st,window):
    return window[st].Get()
def paramVals(event,values,window):
    na = event.split('__')[0]
    if isParam(na):
        defa,defaultKey = getDefa(na)
        if defaultKey == event:
            window[na].update(value=defa)
        elif event == na:
            window[defaultKey].update(value=boolDefa(values,na))
            if event in ['best_of','n']:
                n_Val,b_of_val= getCurrVal(values,'n'),getCurrVal(values,'best_of')
                if n_Val>=b_of_val:
                    window['n'].update(value=int(b_of_val-1))
        elif '_info' in event:
            sg.popup_scrolled(getFullParamDesc(na),keep_on_top=False)
        elif 'disable' in event:
            winDisable(boolIt,na,window)
def catUpdate(event,values,window):
    if 'category_' in event:
        updateCat(event.split('category_')[1])
        promptVars = updateCat(cat)
        updateGuiDispVars()


def _github_issue_help():
    heading_font = '_ 12 bold underline'
    text_font = '_ 10'

    def HelpText(text):
        return sg.Text(text, size=(80, None), font=text_font)

    help_why = \
""" Let's start with a review of the Goals of the PySimpleGUI project
1. To have fun
2. For you to be successful
This form is as important as the documentation and the demo programs to meeting those goals.
The GitHub Issue GUI is here to help you more easily log issues on the PySimpleGUI GitHub Repo. """

    help_goals = \
""" The goals of using GitHub Issues for PySimpleGUI question, problems and suggestions are:
* Give you direct access to engineers with the most knowledge of PySimpleGUI
* Answer your questions in the most precise and correct way possible
* Provide the highest quality solutions possible
* Give you a checklist of things to try that may solve the problem
* A single, searchable database of known problems and their workarounds
* Provide a place for the PySimpleGUI project to directly provide support to users
* A list of requested enhancements
* An easy to use interface to post code and images
* A way to track the status and have converstaions about issues
* Enable multiple people to help users """

    help_explain = \
""" GitHub does not provide a "form" that normal bug-tracking-databases provide. As a result, a form was created specifically for the PySimpleGUI project.
The most obvious questions about this form are
* Why is there a form? Other projects don't have one?
* My question is an easy one, why does it still need a form?
The answer is:
I want you to get your question answered with the highest quality answer possible as quickly as possible.
The longer answer - For quite a while there was no form. It resulted the same back and forth, multiple questions comversation.  "What version are you running?"  "What OS are you using?"  These waste precious time.
If asking nicely helps... PLEASE ... please fill out the form.
I can assume you that this form is not here to punish you. It doesn't exist to make you angry and frustrated.  It's not here for any purpose than to try and get you support and make PySimpleGUI better. """

    help_experience = \
""" Not many Bug-tracking systems ask about you as a user. Your experience in programming, programming in Python and programming a GUI are asked to provide you with the best possible answer.  Here's why it's helpful.  You're a human being, with a past, and a some amount of experience.  Being able to taylor the reply to your issue in a way that fits you and your experience will result in a reply that's efficient and clear.  It's not something normally done but perhaps it should be. It's meant to provide you with a personal response.
If you've been programming for a month, the person answering your question can answer your question in a way that's understandable to you.  Similarly, if you've been programming for 20 years and have used multiple Python GUI frameworks, then you are unlikely to need as much explanation.  You'll also have a richer GUI vocabularly. It's meant to try and give you a peronally crafted response that's on your wavelength. Fun & success... Remember those are our shared goals"""

    help_steps = \
""" The steps to log an issue are:
1. Fill in the form
2. Click Post Issue """
    layout = [
                [sg.T('Goals', font=heading_font, pad=(0,0))],
                [HelpText(help_goals)],
                [sg.T('Why?', font=heading_font, pad=(0,0))],
                [HelpText(help_why)],
                [sg.T('FAQ', font=heading_font, pad=(0,0))],
                [HelpText(help_explain)],
                [sg.T('Experience (optional)', font=heading_font)],
                [HelpText(help_experience)],
                [sg.T('Steps', font=heading_font, pad=(0,0))],
                [HelpText(help_steps)],
                [sg.B('Close')]
              ]
    sg.Window('GitHub Issue GUI Help', layout, keep_on_top=True).read(close=True)

    return

def main_open_github_issue():
    font_frame = '_ 14'
    issue_types = ('Question', 'Bug', 'Enhancement', 'Error Message')
    # frame_type = [[sg.Radio('Question', 1, size=(10,1), enable_events=True, k='-TYPE: QUESTION-'),
    #               sg.Radio('Bug', 1, size=(10,1), enable_events=True, k='-TYPE: BUG-')],
    #              [sg.Radio('Enhancement', 1, size=(10,1), enable_events=True, k='-TYPE: ENHANCEMENT-'),
    #               sg.Radio('Error Message', 1, size=(10,1), enable_events=True, k='-TYPE: ERROR`-')]]
    frame_type = [[sg.Radio(t, 1, size=(10,1), enable_events=True, k=t)] for t in issue_types]

    v_size = (15,1)
    frame_versions = [[sg.T('Python', size=v_size), sg.In(sg.sys.version, size=(20,1), k='-VER PYTHON-')],
                      [sg.T('PySimpleGUI', size=v_size), sg.In(sg.ver, size=(20,1), k='-VER PSG-')],
                      [sg.T('tkinter', size=v_size), sg.In(sg.tclversion_detailed, size=(20,1), k='-VER TK-')],]

    frame_platforms = [ [sg.T('OS                 '), sg.T('Details')],
                        [sg.Radio('Windows', 2, sg.running_windows(), size=(8,1), k='-OS WIN-'), sg.In(size=(8,1),k='-OS WIN VER-')],
                        [sg.Radio('Linux', 2,sg.running_linux(), size=(8,1), k='-OS LINUX-'), sg.In(size=(8,1),k='-OS LINUX VER-')],
                        [sg.Radio('Mac', 2, sg.running_mac(), size=(8,1), k='-OS MAC-'), sg.In(size=(8,1),k='-OS MAC VER-')],
                        [sg.Radio('Other', 2, size=(8,1), k='-OS OTHER-'), sg.In(size=(8,1),k='-OS OTHER VER-')],
                        ]


    frame_experience = [[sg.T('Optional Experience Info')],
                        [sg.In(size=(4,1), k='-EXP PROG-'), sg.T('Years Programming')],
                        [sg.In(size=(4,1), k='-EXP PYTHON-'), sg.T('Years Writing Python')],
                        [sg.CB('Previously programmed a GUI', k='-CB PRIOR GUI-')],
                        [sg.T('Share more if you want....')],
                        [sg.In(size=(25,1), k='-EXP NOTES-')]]

    checklist = (
                  ('Searched main docs for your problem', 'www.PySimpleGUI.org'),
                  ('Looked for Demo Programs that are similar to your goal ', 'http://Demos.PySimpleGUI.org'),
                  ('If not tkinter - looked for Demo Programs for specific port', ''),
                  ('For non tkinter - Looked at readme for your specific port if not PySimpleGUI (Qt, WX, Remi)', ''),
                  ('Run your program outside of your debugger (from a command line)', ''),
                  ('Searched through Issues (open and closed) to see if already reported', 'http://Issues.PySimpleGUI.org'),
                  ('Tried using the PySimpleGUI.py file on GitHub. Your problem may have already been fixed vut not released.', ''))

    frame_checklist = [[sg.CB(c, k=('-CB-', i)), sg.T(t, k='-T{}-'.format(i), enable_events=True)] for i, (c, t) in enumerate(checklist)]

    frame_details = [[sg.Multiline(size=(65,10), font='Courier 10', k='-ML DETAILS-')]]
    frame_code = [[sg.Multiline(size=(80,10), font='Courier 8',  k='-ML CODE-')]]
    frame_markdown = [[sg.Multiline(size=(80,10), font='Courier 8',  k='-ML MARKDOWN-')]]

    top_layout = [  [sg.Col([[sg.Text('Open A GitHub Issue (* = Required Info)', font='_ 15')]], expand_x=True),
                     sg.Col([[sg.B('Help')]])],
                [sg.Frame('Title *', [[sg.Input(k='-TITLE-', size=(50,1), font='_ 14', focus=True)]], font=font_frame)],
                sg.vtop([
                            sg.Frame('Platform *',frame_platforms, font=font_frame),
                            sg.Frame('Type of Issue *',frame_type, font=font_frame),
                            sg.Frame('Versions *',frame_versions, font=font_frame),
                            sg.Frame('Experience',frame_experience, font=font_frame),
                    ]),
                [sg.Frame('Checklist * (note that you can click the links)',frame_checklist, font=font_frame)],
                [sg.HorizontalSeparator()],
                [sg.T(sg.SYMBOL_DOWN + ' If you need more room for details grab the dot and drag to expand', background_color='red', text_color='white')]]

    bottom_layout = [
                [sg.TabGroup([[sg.Tab('Details', frame_details), sg.Tab('Code', frame_code), sg.Tab('Markdown', frame_markdown)]], k='-TABGROUP-')],
                # [sg.Frame('Details',frame_details, font=font_frame, k='-FRAME DETAILS-')],
                # [sg.Frame('Minimum Code to Duplicate',frame_code, font=font_frame, k='-FRAME CODE-')],
                [sg.Text(size=(12,1), key='-OUT-')],
                ]

    layout_pane = sg.Pane([sg.Col(top_layout), sg.Col(bottom_layout)], key='-PANE-')

    layout = [[layout_pane],
              [sg.Col([[sg.B('Post Issue'), sg.B('Create Markdown Only'), sg.B('Quit')]], expand_x=False, expand_y=False)]]

    window = sg.Window('Open A GitHub Issue', layout, finalize=True, resizable=True, enable_close_attempted_event=False)
    for i in range(len(checklist)):
        window['-T{}-'.format(i)].set_cursor('hand1')
    window['-TABGROUP-'].expand(True, True, True)
    window['-ML CODE-'].expand(True, True, True)
    window['-ML DETAILS-'].expand(True, True, True)
    window['-ML MARKDOWN-'].expand(True, True, True)
    window['-PANE-'].expand(True, True, True)
    # window['-FRAME CODE-'].expand(True, True, True)
    # window['-FRAME DETAILS-'].expand(True, True, True)

    while True:             # Event Loop
        event, values = window.read()
        # print(event, values)
        if event in (sg.WINDOW_CLOSE_ATTEMPTED_EVENT, 'Quit'):
            if sg.popup_yes_no( 'Do you really want to exit?',
                                'If you have not clicked Post Issue button and then clicked "Submit New Issue" button '
                                'then your issue will not have been submitted to GitHub.'
                                'Do no exit until you have PASTED the information from Markdown tab into an issue?') == 'Yes':
                break
        if event == sg.WIN_CLOSED:
            break
        if event in ['-T{}-'.format(i) for i in range(len(checklist))]:
            webbrowser.open_new_tab(window[event].get())
        if event in issue_types:
            title = str(values['-TITLE-'])
            if len(title) != 0:
                if title[0] == '[' and title.find(']'):
                    title = title[title.find(']')+1:]
                    title = title.strip()
            window['-TITLE-'].update('[{}] {}'.format(event, title))
        if event == 'Help':
            _github_issue_help()
        elif event in ('Post Issue', 'Create Markdown Only'):
            issue_type = None
            for itype in issue_types:
                if values[itype]:
                    issue_type = itype
                    break
            if issue_type is None:
                sg.popup_error('Must choose issue type')
                continue
            if values['-OS WIN-']:
                operating_system = 'Windows'
                os_ver = values['-OS WIN VER-']
            elif values['-OS LINUX-']:
                operating_system = 'Linux'
                os_ver = values['-OS LINUX VER-']
            elif values['-OS MAC-']:
                operating_system = 'Mac'
                os_ver = values['-OS MAC VER-']
            elif values['-OS OTHER-']:
                operating_system = 'Other'
                os_ver = values['-OS OTHER VER-']
            else:
                sg.popup_error('Must choose Operating System')
                continue
            checkboxes = ['X' if values[('-CB-', i)] else ' ' for i in range(len(checklist))]

            if not _github_issue_post_validate(values, checklist, issue_types):
                continue

            markdown = _github_issue_post_make_markdown(issue_type, operating_system, os_ver, 'tkinter', values['-VER PSG-'], values['-VER TK-'], values['-VER PYTHON-'],
                                                        values['-EXP PYTHON-'], values['-EXP PROG-'],  'Yes' if values['-CB PRIOR GUI-'] else 'No', values['-EXP NOTES-'], *checkboxes, values['-ML DETAILS-'], values['-ML CODE-'])
            window['-ML MARKDOWN-'].update(markdown)
            link = _github_issue_post_make_github_link(values['-TITLE-'], window['-ML MARKDOWN-'].get())
            if event == 'Post Issue':
                webbrowser.open_new_tab(link)
            else:
                sg.popup('Your markdown code is in the Markdown tab', keep_on_top=True)

    window.close()

if __name__ == '__main__':
    # sg.theme(sg.OFFICIAL_PYSIMPLEGUI_THEME)
    main_open_github_issue()
