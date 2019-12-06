import copy
from multiprocessing import Process, Event, Pipe, Queue

from threading import Thread
import PySimpleGUI as sg
import urllib.request

from pydub import AudioSegment
import io
from classifier import Classifier
from functions import get_input_vector
import numpy as np
from queue import Empty
from http.client import BadStatusLine
from urllib.error import URLError

FRAME_WIDTH = 441
NUM_FRAMES = 100
FRQ = 22050


class MyAudioSegment(AudioSegment):
    @classmethod
    def from_bytes(cls, b, f):
        segment = AudioSegment.from_file(io.BytesIO(b), format=f)
        segment = segment.set_channels(1).set_frame_rate(FRQ).set_sample_width(2)
        return segment

    def get_array_of_samples(self, *args, **kwargs):
        return np.array(super().get_array_of_samples(*args, **kwargs))


def decode_to_url(link):
    if link.startswith('http') and not link.endswith(('.pls', '.m3u')):
        return link
    if link.startswith('http'):
        open_fun = urllib.request.urlopen
        line_proc = lambda x: x.decode("utf-8")
    else:
        open_fun = open
        line_proc = lambda x: x
    if link.endswith('.m3u'):
        with open_fun(link) as file:
            for line in file:
                line = line_proc(line)

                if not line.startswith('#') and len(line) > 1:
                    return line
    if link.endswith('.pls'):
        with open_fun(link) as file:
            for line in file:
                line = line_proc(line)
                if 'http' in line:
                    return line.split('=')[1]


def get_format(url_conn):
    content_type = url_conn.getheader('Content-Type')
    if(content_type == 'audio/mpeg'):
        return 'mp3'
    elif(content_type == 'application/aacp' or content_type == 'audio/aacp'):
        return 'aac'
    elif(content_type == 'application/ogg' or content_type == 'audio/ogg'):
        return 'ogg'
    else:
        print('Nieobsługiwany format audio: "' + content_type + '".')
        return None


def get_bsize(url_conn):
    bitrate = url_conn.getheader('icy-br')
    if bitrate:
        bitrate = bitrate.split(',')[0]
        b_size = (int(bitrate)) * 125
    else:
        b_size = 16000
    return b_size


def audio_segment_proc(audio_format, bytes_q, input_conn):
    samples = np.array(1, dtype=np.int16)
    while True:
        try:
            bts = bytes_q.get(timeout=5)
        except Empty:
            break
        if bts is None:
            break
        segment = MyAudioSegment.from_bytes(bts, audio_format)
        converted = np.trim_zeros(segment.get_array_of_samples(), 'f')
        samples = np.append(samples, converted)

        if samples.size >= 2*FRQ:
            input_conn.send(get_input_vector(samples[:2*FRQ], FRAME_WIDTH, NUM_FRAMES))
            samples = samples[2*FRQ:]


def radio_proc(input_conn, recv_enable_ev, radio_link):
    radio_url = decode_to_url(radio_link)
    try:
        url_conn = urllib.request.urlopen(radio_url)
    except BadStatusLine:
        return
    except URLError:
        return

    audio_format = get_format(url_conn)
    if not audio_format:
        return

    bsize = get_bsize(url_conn)

    bytes_q = Queue()

    audio_segment_thread = Thread(target=audio_segment_proc,
                                  args=(audio_format, bytes_q, input_conn))
    audio_segment_thread.start()

    while (not url_conn.closed and recv_enable_ev.is_set()):
        bytes_q.put(url_conn.read(bsize))

    bytes_q.put(None)


def classifier(input_conn, output_send_conn, start_event):

    clf = Classifier()
    start_event.set()

    while True:
        if input_conn.poll():
            x_v = input_conn.recv()

            prediction = clf.predict(x_v)
            output_send_conn.send(prediction)
        if not start_event.is_set():
            break


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

    input_recv_conn, input_send_conn = Pipe(duplex=False)
    output_recv_conn, output_send_conn = Pipe(duplex=False)
    keras_start_ev = Event()
    recv_enable_ev = Event()
    keras_start_ev.clear()
    recv_enable_ev.clear()

    classifier_process = Process(target=classifier,
                               args=(input_recv_conn, output_send_conn, keras_start_ev, ))
    radio_process = Process()

    classifier_process.start()
    keras_start_ev.wait()

    def _proc_stop():
        recv_enable_ev.clear()
        radio_process.join()
        start_stop_btn.Update('Start')

    def _proc_start():
        recv_enable_ev.set()
        radio_process.start()
        start_stop_btn.Update('Stop')

    def _show_class(prediction):
        prediction_bar.UpdateBar(prediction)
        prediction_prc.Update(f'{prediction:1.2f}')
        if prediction > 0.5:
            prediction_cls.Update('MUZYKA')
        else:
            prediction_cls.Update('TREŚCI INFORMACYJNE')

    while True:
        if output_recv_conn.poll():
            prediction = output_recv_conn.recv()[0][0]
            _show_class(prediction)

        event, values = window.read(1000)
        if recv_enable_ev.is_set() and not radio_process.is_alive():
            _proc_stop()

        if event == 'switch':
            if not radio_process.is_alive():
                radio_process = Process(target=radio_proc, args=(input_send_conn, recv_enable_ev, values['link']))
                _proc_start()
            else:
                _proc_stop()

        if (not event and not values) or event == 'close':
            break

    if radio_process.is_alive():
        _proc_stop()

    keras_start_ev.clear()
    classifier_process.join()
    classifier_process.close()


if __name__ == '__main__':
    main()
