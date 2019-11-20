import PySimpleGUI as sg

sg.change_look_and_feel('DarkAmber')

layout = [[sg.Text('Enter 2 files to comare')],
          [sg.Text('Z adresu', size=(8, 1)), sg.Input(key='address')],
          [sg.Text('Z pliku', size=(8, 1)), sg.Input(key='file'),
           sg.FileBrowse(button_text='Przeglądaj', file_types=(('M3U', '*.m3u'), ('PLS', '*.*')))],
          [sg.Submit(), sg.Cancel()]]

window = sg.Window('File Compare', layout)

event, values = window.read()
window.close()
print(f'You clicked {event}')
print(values)
print(f'You chose filenames {values[0]} and {values[1]}')