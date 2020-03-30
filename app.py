from http.client import BadStatusLine
from multiprocessing import Process, Event, Pipe, Queue
from queue import Empty
from threading import Thread
from urllib.error import URLError

import PySimpleGUI as sg
import numpy as np
import urllib.request

from classifier.classifier import Classifier
from classifier.functions import get_input_vector
from classifier.audio_segment import MyAudioSegment
from classifier.config import FRAME_WIDTH, NUM_FRAMES_WINDOW, FREQUENCY


def decode_to_url(link: str) -> str:
    """
    Returns the URL of radio station extracted from `link` to the file or website.
    """
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
                    return line.strip()
    if link.endswith('.pls'):
        with open_fun(link) as file:
            for line in file:
                line = line_proc(line)
                if 'http' in line:
                    return line.split('=')[1].strip()


def get_format(url_conn: object) -> str:
    """
    Returns format of received audio data

    :param url_conn: urlopen
    :return: format
    """
    content_type = url_conn.getheader('Content-Type')
    if content_type == 'audio/mpeg':
        return 'mp3'
    elif content_type == 'application/aacp' or content_type == 'audio/aacp':
        return 'aac'
    elif content_type == 'application/ogg' or content_type == 'audio/ogg':
        return 'ogg'
    else:
        print('Audio format "' + content_type + '" not supported.')
        return ''


def get_bsize(url_conn: object) -> int:
    """
    Returns size of received packets

    :param url_conn: urlopen
    """
    bitrate = url_conn.getheader('icy-br')
    if bitrate:
        bitrate = bitrate.split(',')[0]
        b_size = (int(bitrate)) * 125
    else:
        b_size = 16000
    return b_size


def audio_segment_thrd(audio_format, bytes_q, input_conn):
    """
    Converts received data to a classifier input vector and sends it to the classifiers process
    """
    samples = np.array(1, dtype=np.int16)
    while True:
        try:
            bts = bytes_q.get(timeout=5)
        except Empty:
            break
        if bts is None:
            break
        segment = MyAudioSegment.from_bytes(bts, audio_format, FREQUENCY)
        converted = np.trim_zeros(segment.get_ndarray_of_samples(), 'f')
        samples = np.append(samples, converted)

        # if the segment is longer than 2s
        if samples.size >= 2*FREQUENCY:
            input_conn.send(get_input_vector(samples[:2*FREQUENCY], FRAME_WIDTH, NUM_FRAMES_WINDOW))
            samples = samples[2*FREQUENCY:]


def radio_proc(input_conn: object, recv_enable_ev: object, radio_link: str):
    """
    Handles receiving and analyzing data from a radio station at `radio_link`.
    """
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

    audio_segment_thread = Thread(target=audio_segment_thrd,
                                  args=(audio_format, bytes_q, input_conn))
    audio_segment_thread.start()

    while (not url_conn.closed and recv_enable_ev.is_set()):
        bytes_q.put(url_conn.read(bsize))

    bytes_q.put(None)


def classifier_proc(input_conn, output_send_conn, start_event):
    """
    Receives input vector from `input_conn` and make the prediction which is send by `output_send_conn`
    """
    clf = Classifier()
    start_event.set()

    while True:
        if input_conn.poll():
            x_v = input_conn.recv()

            prediction = clf.predict(x_v)
            output_send_conn.send(prediction)
        if not start_event.is_set():
            break


def run():
    sg.change_look_and_feel('DarkBlue1')
    layout = [
        [
            sg.Text('Link', size=(6, 1)), sg.Input(size=(36, 1), key='link'),
            sg.FileBrowse(size=(8, 1), button_text='Open', file_types=(('M3U', '*.m3u'), ('PLS', '*.pls')))
        ],
        [
            sg.Text('Result', size=(6, 1)),
            sg.ProgressBar(1, orientation='h', size=(8, 20), key='prediction_bar'),
            sg.Text('-,--', size=(6, 1), key='prediction_prc'),
            sg.Text('-', size=(20, 1), key='prediction_cls')
        ],
        [
            sg.Button(button_text='Start', key='switch'),
            sg.CloseButton(button_text='Exit'),
        ]
    ]
    window = sg.Window('Classifier', layout)
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

    classifier_process = Process(target=classifier_proc,
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
            prediction_cls.Update('MUSIC')
        else:
            prediction_cls.Update('NOT MUSIC')

    while True:
        if output_recv_conn.poll():
            prediction = output_recv_conn.recv()[0][0]
            _show_class(prediction)

        event, values = window.read(1000)
        if recv_enable_ev.is_set() and not radio_process.is_alive():
            _proc_stop()

        if event == 'switch':
            if not values['link']:
                pass
            elif not radio_process.is_alive():
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
    run()
