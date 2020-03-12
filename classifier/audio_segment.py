from pydub import AudioSegment
import io
import numpy as np


class MyAudioSegment(AudioSegment):
    @classmethod
    def from_bytes(cls, bts: bytes, format: str, freq:int) -> AudioSegment:
        """
        Returns mono AudioSegment of the audio file described by raw data `bytes` in `format` format.
        """
        segment = cls.from_file(io.BytesIO(bts), format=format)
        segment = segment.set_channels(1).set_frame_rate(freq).set_sample_width(2)
        return segment

    @classmethod
    def from_wav(cls, file, parameters=None):
        segment = cls.from_file(file, 'wav', parameters=parameters)
        return segment

    def get_ndarray_of_samples(self, *args: object) -> np.ndarray:
        return np.array(self.get_array_of_samples())
