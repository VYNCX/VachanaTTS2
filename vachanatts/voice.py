import itertools
import json
import logging
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime

from .config import Config, SpeechConfig
from .phoneme_ids import phonemes_to_ids, BOS, EOS, PAD
from .th_phonemizer import phonemizer

_DEFAULT_SYNTHESIS_CONFIG = SpeechConfig()
_MAX_WAV_VALUE = 32767.0
_PHONEME_BLOCK_PATTERN = re.compile(r"(\[\[.*?\]\])")

_LOGGER = logging.getLogger(__name__)

@dataclass
class PhonemeAlignment:
    phoneme: str
    phoneme_ids: Sequence[int]
    num_samples: int

@dataclass
class AudioChunk:
    """Chunk of raw audio."""

    sample_rate: int
    """Rate of chunk samples in Hertz."""

    sample_width: int
    """Width of chunk samples in bytes."""

    sample_channels: int
    """Number of channels in chunk samples."""

    audio_float_array: np.ndarray
    """Audio data as float numpy array in [-1, 1]."""

    phonemes: list[str]
    """Phonemes that produced this audio chunk."""

    phoneme_ids: list[int]
    """Phoneme ids that produced this audio chunk."""

    phoneme_id_samples: Optional[np.ndarray] = None
    """Number of audio samples for each phoneme id (alignments).

    Only available for supported voice models.
    """

    phoneme_alignments: Optional[list[PhonemeAlignment]] = None
    """Alignments between phonemes and audio samples."""

    # ---

    _audio_int16_array: Optional[np.ndarray] = None
    _audio_int16_bytes: Optional[bytes] = None
    _phoneme_alignments: Optional[list[PhonemeAlignment]] = None

    @property
    def audio_int16_array(self) -> np.ndarray:
        """
        Get audio as an int16 numpy array.

        :return: Audio data as int16 numpy array.
        """
        if self._audio_int16_array is None:
            self._audio_int16_array = np.clip(
                self.audio_float_array * _MAX_WAV_VALUE, -_MAX_WAV_VALUE, _MAX_WAV_VALUE
            ).astype(np.int16)

        return self._audio_int16_array

    @property
    def audio_int16_bytes(self) -> bytes:
        """
        Get audio as 16-bit PCM bytes.

        :return: Audio data as signed 16-bit sample bytes.
        """
        return self.audio_int16_array.tobytes()


@dataclass
class Voice:
    """A voice for Piper."""

    session: onnxruntime.InferenceSession
    """ONNX session."""

    config: Config
    """Piper voice configuration."""

    """Path to espeak-ng data directory."""

    @staticmethod
    def load(
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        use_cuda: bool = False
    ) -> "Voice":
        """
        Load an ONNX model and config.

        :param model_path: Path to ONNX voice model.
        :param config_path: Path to JSON voice config (defaults to model_path + ".json").
        :param use_cuda: True if CUDA (GPU) should be used instead of CPU.
        :param espeak_data_dir: Path to espeak-ng data dir (defaults to internal data).
        :return: Voice object.
        """
        if config_path is None:
            config_path = f"{model_path}.json"
            _LOGGER.debug("Guessing voice config path: %s", config_path)

        with open(config_path, "r", encoding="utf-8") as config_file:
            config_dict = json.load(config_file)

        providers: list[Union[str, tuple[str, dict[str, Any]]]]
        if use_cuda:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {"cudnn_conv_algo_search": "HEURISTIC"},
                )
            ]
            _LOGGER.debug("Using CUDA")
        else:
            providers = ["CPUExecutionProvider"]

        return Voice(
            config=Config.from_dict(config_dict),
            session=onnxruntime.InferenceSession(
                str(model_path),
                sess_options=onnxruntime.SessionOptions(),
                providers=providers,
            )
        )

    def phonemes_to_ids(self, phonemes: list[str]) -> list[int]:
        """
        Phonemes to ids.

        :param phonemes: List of phonemes.
        :return: List of phoneme ids.
        """
        return phonemes_to_ids(phonemes, self.config.phoneme_id_map)

    def synthesize(
        self,
        text: str,
        syn_config: Optional[SpeechConfig] = None,
        include_alignments: bool = False
    ) -> Iterable[AudioChunk]:
        """
        Synthesize one audio chunk per sentence from from text.

        :param text: Text to synthesize.
        :param syn_config: Synthesis configuration.
        :param include_alignments: If True and the model supports it, include phoneme/audio alignments.
        """
        if syn_config is None:
            syn_config = _DEFAULT_SYNTHESIS_CONFIG

        sentence_phonemes = phonemizer(text)
        #print(sentence_phonemes)
        _LOGGER.debug("text=%s, phonemes=%s", text, sentence_phonemes)

        for phonemes in sentence_phonemes:
            if not phonemes:
                continue

            phoneme_ids = self.phonemes_to_ids(phonemes)

            phoneme_id_samples: Optional[np.ndarray] = None
            audio_result = self.phoneme_ids_to_audio(
                phoneme_ids, syn_config, include_alignments=include_alignments
            )
            if isinstance(audio_result, tuple):
                # Audio + alignments
                audio, phoneme_id_samples = audio_result
            else:
                # Audio only
                audio = audio_result

            if syn_config.normalize_audio:
                max_val = np.max(np.abs(audio))
                if max_val < 1e-8:
                    # Prevent division by zero
                    audio = np.zeros_like(audio)
                else:
                    audio = audio / max_val

            if syn_config.volume != 1.0:
                audio = audio * syn_config.volume

            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

            phoneme_alignments: Optional[list[PhonemeAlignment]] = None
            if (phoneme_id_samples is not None) and (
                len(phoneme_id_samples) == len(phoneme_ids)
            ):

                pad_ids = self.config.phoneme_id_map.get(PAD, [])
                phoneme_id_idx = 0
                phoneme_alignments = []
                alignment_failed = False
                for phoneme in itertools.chain([BOS], phonemes, [EOS]):
                    expected_ids = self.config.phoneme_id_map.get(phoneme, [])

                    if phoneme != EOS:
                        ids_to_check = list(itertools.chain(expected_ids, pad_ids))
                    else:
                        ids_to_check = expected_ids

                    start_phoneme_id_idx = phoneme_id_idx
                    for phoneme_id in ids_to_check:
                        if phoneme_id_idx >= len(phoneme_ids):
                            # Ran out of phoneme ids
                            alignment_failed = True
                            break

                        if phoneme_id != phoneme_ids[phoneme_id_idx]:
                            # Bad alignment
                            alignment_failed = True
                            break

                        phoneme_id_idx += 1

                    if alignment_failed:
                        break

                    phoneme_alignments.append(
                        PhonemeAlignment(
                            phoneme=phoneme,
                            phoneme_ids=ids_to_check,
                            num_samples=sum(
                                phoneme_id_samples[start_phoneme_id_idx:phoneme_id_idx]
                            ),
                        )
                    )

                if alignment_failed:
                    phoneme_alignments = None
                    _LOGGER.debug("Phoneme alignment failed")

            yield AudioChunk(
                sample_rate=self.config.sample_rate,
                sample_width=2,
                sample_channels=1,
                audio_float_array=audio,
                phonemes=phonemes,
                phoneme_ids=phoneme_ids,
                phoneme_id_samples=phoneme_id_samples,
                phoneme_alignments=phoneme_alignments,
            )

    def synthesize_wav(
        self,
        text: str,
        wav_file: wave.Wave_write,
        syn_config: Optional[SpeechConfig] = None,
        set_wav_format: bool = True,
        include_alignments: bool = False
    ) -> Optional[list[PhonemeAlignment]]:
        """
        Synthesize and write WAV audio from text.

        :param text: Text to synthesize.
        :param wav_file: WAV file writer.
        :param syn_config: Synthesis configuration.
        :param set_wav_format: True if the WAV format should be set automatically.
        :param include_alignments: If True and the model supports it, return phoneme/audio alignments.

        :return: Phoneme/audio alignments if include_alignments is True, otherwise None.
        """
        alignments: list[PhonemeAlignment] = []
        first_chunk = True
        for audio_chunk in self.synthesize(
            text, syn_config=syn_config, include_alignments=include_alignments
        ):
            if first_chunk:
                if set_wav_format:
                    # Set audio format on first chunk
                    wav_file.setframerate(audio_chunk.sample_rate)
                    wav_file.setsampwidth(audio_chunk.sample_width)
                    wav_file.setnchannels(audio_chunk.sample_channels)

                first_chunk = False

            wav_file.writeframes(audio_chunk.audio_int16_bytes)

            if include_alignments and audio_chunk.phoneme_alignments:
                alignments.extend(audio_chunk.phoneme_alignments)

        if include_alignments:
            return alignments

        return None

    def phoneme_ids_to_audio(
        self,
        phoneme_ids: list[int],
        syn_config: Optional[SpeechConfig] = None,
        include_alignments: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Synthesize raw audio from phoneme ids.

        :param phoneme_ids: List of phoneme ids.
        :param syn_config: Synthesis configuration.
        :param include_alignments: Return samples per phoneme id if True.
        :return: Audio float numpy array from voice model (unnormalized, in range [-1, 1]).

        If include_alignments is True and the voice model supports it, the return
        value will be a tuple instead with (audio, phoneme_id_samples) where
        phoneme_id_samples contains the number of audio samples per phoneme id.
        """
        if syn_config is None:
            syn_config = _DEFAULT_SYNTHESIS_CONFIG

        speaker_id = syn_config.speaker_id
        length_scale = syn_config.length_scale
        noise_scale = syn_config.noise_scale
        noise_w_scale = syn_config.noise_w_scale

        if length_scale is None:
            length_scale = self.config.length_scale

        if noise_scale is None:
            noise_scale = self.config.noise_scale

        if noise_w_scale is None:
            noise_w_scale = self.config.noise_w_scale

        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
        scales = np.array(
            [noise_scale, length_scale, noise_w_scale],
            dtype=np.float32,
        )

        args = {
            "input": phoneme_ids_array,
            "input_lengths": phoneme_ids_lengths,
            "scales": scales,
        }

        if self.config.num_speakers <= 1:
            speaker_id = None

        if (self.config.num_speakers > 1) and (speaker_id is None):
            # Default speaker
            speaker_id = 0

        if speaker_id is not None:
            sid = np.array([speaker_id], dtype=np.int64)
            args["sid"] = sid

        # Synthesize through onnx
        result = self.session.run(
            None,
            args,
        )
        audio = result[0].squeeze()
        if not include_alignments:
            return audio

        if len(result) == 1:
            # Alignment is not available from voice model
            return audio, None

        # Number of samples for each phoneme id
        phoneme_id_samples = (result[1].squeeze() * self.config.hop_length).astype(
            np.int64
        )

        return audio, phoneme_id_samples
