import copy
from multiprocessing import Process, Event, Pipe

from threading import Thread
from multiprocessing.connection import wait
import PySimpleGUI as sg
import urllib.request
from pydub import AudioSegment
import io
from functions import zcr_ste, zcr_diff_mean, zcr_third_central_moment, zcr_exceed_th, zcr_std_of_fod, ste_mler
import numpy as np
from pydub.playback import play
from queue import Queue

FRAME_WIDTH = 450
NUM_FRAMES = 100
FRQ = 22050
KILL_MSG = 'order66'

# class MyAudioSegment(AudioSegment):


def decode_to_url(link):
    if link.startswith('http') and not link.endswith(('.pls', '.m3u')):
        return link

    if link.startswith('http'):
        open_fun = urllib.request.urlopen
    else:
        open_fun = open
    if link.endswith('.m3u'):
        with open_fun(link) as file:
            for line in file:
                if not line.startswith('#') and len(line) > 1:
                    return line
    if link.endswith('.pls'):
        with open_fun(link) as remotefile:
            for line in remotefile:
                if not line.startswith('#') and len(line) > 1:
                    return line



    # return 'http://br-brklassik-live.cast.addradio.de/br/brklassik/live/mp3/128/stream.mp3'


def get_format(url_conn):
    try:
        eh = url_conn.getheader('icy-br')
        print(eh)
    except:
        pass

    content_type = url_conn.getheader('Content-Type')
    if(content_type == 'audio/mpeg'):
        return 'mp3'
    elif(content_type == 'application/aacp' or content_type == 'audio/aacp'):
        return 'aac'
    elif(content_type == 'application/ogg' or content_type == 'audio/ogg'):
        return 'ogg'
    elif(content_type == 'audio/x-mpegurl'):
        print('Sorry, M3U playlists are currently not supported')
    else:
        print('Unknown content type "' + content_type + '". Assuming mp3.')
        return '.mp3'




# def audio_segment_process(audio_format, data_conn, input_conn):
#     segment = AudioSegment.empty()
#     while True:
#         try:
#             bytes = data_conn.recv()
#         except EOFError as e:
#             print(e)
#             break
#
#         segment = segment + AudioSegment.from_file(io.BytesIO(bytes), format=audio_format)
#         if segment.duration_seconds >= 2:
#             window = segment[:2000]
#             window = window.set_channels(1)
#             window = window.set_frame_rate(22500)
#             window.export('jezu.wav', format='wav')
#
#             window_v = aud_seg_to_array(window)
#             input_conn.send(get_input_vector(window_v))
#             segment = segment[2000:]
def del_silence(sound, silence_threshold=-200.0, chunk_size=10):
    trim_start, trim_end = 0, 0
    while sound[trim_start:trim_start+chunk_size].dBFS < silence_threshold and trim_start < len(sound):
        trim_start += chunk_size
    while sound.reverse()[trim_end:trim_end + chunk_size].dBFS < silence_threshold and trim_end < len(sound):
        trim_end += chunk_size

    return sound[trim_start:-trim_end]


def audio_segment_process(audio_format, bytes_q, input_conn):
    segment = AudioSegment.empty()
    while True:
        bytes = bytes_q.get(timeout=5)
        if bytes is None:
            break
        print(len(bytes))
        converted = AudioSegment.from_file(io.BytesIO(bytes), format=audio_format)
        segment = segment + converted
        print(segment.duration_seconds)
        if segment.duration_seconds >= 2:
            window = segment[:2000]
            if window.channels != 1:
                window = window.set_channels(1)
            if window.sample_width != 2:
                window.set_sample_width(2)
            if window.frame_rate != FRQ:
                window = window.set_frame_rate(FRQ)
            window.export('jezu.wav', format='wav')
            print('siup')
            window_v = aud_seg_to_array(window)
            print('ehh')
            input_conn.send(get_input_vector(window_v))
            # segment = segment[2000:]
            segment = AudioSegment.empty()


def read_transmission(input_conn, radio_recv_ev, radio_link):
    radio_url = decode_to_url(radio_link)
    url_conn = urllib.request.urlopen(radio_url)
    audio_format = get_format(url_conn)
    # segment = AudioSegment.empty()
    bytes_q = Queue()

    audio_segmrnt_thread = Thread(target=audio_segment_process,
                                  args=(audio_format, bytes_q, input_conn ))
    audio_segmrnt_thread.start()

    # data_recv_conn, data_send_conn = Pipe(duplex=False)
    # audio_segment_proc = Process(target=audio_segment_process,
    #                              args=(audio_format, data_recv_conn, input_conn ))

    # audio_segment_proc.start()

    while (not url_conn.closed and radio_recv_ev.is_set()):
        bytes_q.put(url_conn.read(10240))
        # bytes_q.put(bytes)
        # segment = segment + AudioSegment.from_file(io.BytesIO(bytes), format=audio_format)
        # if segment.duration_seconds >= 2:
        #     window = segment[:2000]
        #     window = window.set_channels(1)
        #     window = window.set_frame_rate(22500)
        #     window.export('jezu.wav', format='wav')
        #
        #     window_v = aud_seg_to_array(window)
        #     input_conn.send(get_input_vector(window_v))
        #     segment = segment[2000:]
    bytes_q.put(None)
    print('halo?')


def get_input_vector(song):
    zcr_v, ste_v = zcr_ste(np.array(song), FRAME_WIDTH, NUM_FRAMES)
    zcr_mean = zcr_v.mean()
    input_vector = [
        zcr_diff_mean(zcr_v, zcr_mean),
        zcr_third_central_moment(zcr_v),
        zcr_exceed_th(zcr_v, zcr_mean),
        zcr_std_of_fod(zcr_v),
        ste_mler(ste_v),
    ]
    return np.array([input_vector])


def aud_seg_to_array(segment):
    samples = segment.get_array_of_samples()
    return np.array(samples)


def classifier(input_conn, output_send_conn, start_event):
    from tensorflow.python import keras
    import time
    import csv
    model = keras.models.load_model('hit.hdf5')
    f = open(r'krok.csv', 'a')
    start_event.set()
    while True:
        pass
        try:
            x_v = input_conn.recv()
        except EOFError as e:
            print(e)
            break

        if type(x_v) is str and x_v == KILL_MSG:
            break
        else:
            print(x_v)
            prediction = model.predict(x_v)
            print(prediction)

            writer = csv.writer(f)
            writer.writerow([time.time(), prediction])

            output_send_conn.send(prediction)


def main():
    sg.change_look_and_feel('DarkBlue1')

    layout = [
        [
            sg.Text('Stacja', size=(6, 1)), sg.Input(size=(36, 1), key='link'),
            sg.FileBrowse(size=(8, 1), button_text='Otwórz', file_types=(('M3U', '*.m3u'), ('PLS', '*.pls')))
        ],
        [
            sg.Text('Wynik', size=(6, 1)),
            sg.ProgressBar(1, orientation='h', size=(8, 20), key='prediction_bar'),
            sg.Text('-,--', size=(6, 1), key='prediction_prc'),
            sg.Text('-', size=(20, 1), key='prediction_cls')
        ],
        [
            sg.Button(button_text='Start', key='switch'),
            sg.CloseButton(button_text='Zamknij'),

        ]
    ]

    window = sg.Window('Klasyfikator', layout)
    prediction_bar = window['prediction_bar']
    prediction_prc = window['prediction_prc']
    prediction_cls = window['prediction_cls']
    start_stop_btn = window['switch']

    # flow czytania transmisji
    input_recv_conn, input_send_conn = Pipe(duplex=False)
    output_recv_conn, output_send_conn = Pipe(duplex=False)
    keras_start_ev = Event()
    keras_start_ev.clear()
    radio_recv_ev = Event()
    radio_recv_ev.clear()

    classify_process = Process(target=classifier,
                               args=(input_recv_conn, output_send_conn, keras_start_ev, ))
    read_process = Process()

    classify_process.start()
    keras_start_ev.wait()

    while True:
        if output_recv_conn.poll():
            prediction = output_recv_conn.recv()[0][0]
            prediction_bar.UpdateBar(prediction)
            prediction_prc.Update(f'{prediction:1.2f}')
            if prediction > 0.5:
                prediction_cls.Update('MUZYKA')
            else:
                prediction_cls.Update('TREŚCI INFORMACYJNE')

        event, values = window.read(1000)
        if event == 'switch':
            if not read_process.is_alive():
                read_process = Process(target=read_transmission, args=(input_send_conn, radio_recv_ev, values['link']))
                radio_recv_ev.set()
                read_process.start()
                start_stop_btn.Update('Stop')
            else:
                radio_recv_ev.clear()
                read_process.join()
                start_stop_btn.Update('Start')

        if (not event and not values) or event == 'close' :
            break

    radio_recv_ev.clear()
    input_send_conn.close()

    if read_process.is_alive():
        read_process.join()
        read_process.close()

    classify_process.join()
    classify_process.close()


if __name__ == '__main__':
    main()
