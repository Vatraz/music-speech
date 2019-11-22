import PySimpleGUI as sg

sg.change_look_and_feel('DarkAmber')

layout = [[sg.Text('Enter 2 files to comare')],
          [sg.Text('Stacja', size=(6, 1)), sg.Input(size=(36, 1), key='file'),
           sg.FileBrowse(size=(8, 1), button_text='Otwórz', file_types=(('M3U', '*.m3u'), ('PLS', '*.*')))],
          [sg.Submit(), sg.Cancel()]]

window = sg.Window('File Compare', layout)
while True:
    event, values = window.read()
window.close()
print(f'You clicked {event}')
print(values)
print(f'You chose filenames {values[0]} and {values[1]}')