"""Sample rate and container conversion for finished clips.

Gemini TTS has no output-format knob: every request comes back as headerless
16-bit mono PCM at 24 kHz, and the mime type it reports says so. A dataset,
though, usually wants a particular rate — 16 kHz for most ASR fine-tunes, 22.05
kHz for VITS and Piper — and sometimes a smaller container than WAV. So the
conversion happens here, once, right after synthesis, rather than being asked of
an API that cannot do it.

ffmpeg does the work when it is installed (soxr resampling, every codec). Without
it, WAV output is still handled in pure Python, because rate conversion is the
common case and it should not need a system package.
"""
from __future__ import annotations

import array
import os
import shutil
import struct
import subprocess
import tempfile
from typing import Optional, Tuple

# Formats a clip can be written as. Lossless ones ignore `bitrate`.
FORMATS = ("wav", "mp3", "flac", "ogg", "opus")
LOSSLESS = ("wav", "flac")

MIME_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "opus": "audio/ogg",
}

# Opus only encodes at these rates; ffmpeg resamples silently, which would make
# the declared rate a lie, so the caller is told instead.
OPUS_RATES = (8000, 12000, 16000, 24000, 48000)

_WARNED = set()


class AudioError(Exception):
    pass


def ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def validate(fmt: str, sample_rate: Optional[int]) -> None:
    """Fail at config time rather than after a few thousand API calls."""
    if fmt not in FORMATS:
        raise AudioError(f"Unknown audio format {fmt!r}. Choose one of: {', '.join(FORMATS)}")
    if sample_rate is not None and not 4000 <= sample_rate <= 192000:
        raise AudioError(f"audio.sample_rate {sample_rate} is outside 4000-192000 Hz")
    if fmt == "opus" and sample_rate is not None and sample_rate not in OPUS_RATES:
        raise AudioError(
            f"Opus only encodes at {', '.join(str(r) for r in OPUS_RATES)} Hz, not {sample_rate}."
        )
    if fmt != "wav" and not ffmpeg():
        raise AudioError(
            f"Writing {fmt} needs ffmpeg on PATH (apt install ffmpeg / brew install ffmpeg). "
            f"Use audio.format: wav to stay dependency-free."
        )


def parse_wav(data: bytes) -> Tuple[bytes, int, int, int]:
    """(pcm, sample_rate, channels, bits_per_sample) from a RIFF/WAVE file.

    Chunks are walked rather than assumed at fixed offsets: a WAV that came back
    from ffmpeg carries a LIST chunk before the data.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError("not a WAV file")
    rate = channels = bits = 0
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset:offset + 4]
        size = struct.unpack("<I", data[offset + 4:offset + 8])[0]
        body = offset + 8
        if chunk_id == b"fmt " and size >= 16:
            channels, rate = struct.unpack("<HI", data[body + 2:body + 8])
            bits = struct.unpack("<H", data[body + 14:body + 16])[0]
        elif chunk_id == b"data":
            return data[body:body + size], rate, channels, bits
        offset = body + size + (size % 2)
    raise AudioError("WAV file has no data chunk")


def build_wav(pcm: bytes, rate: int, channels: int = 1, bits: int = 16) -> bytes:
    block_align = channels * (bits // 8)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1,
        channels, rate, rate * block_align, block_align, bits,
        b"data", len(pcm),
    )
    return header + pcm


def _downmix(samples: array.array, channels: int) -> array.array:
    if channels <= 1:
        return samples
    out = array.array("h", bytes(2 * (len(samples) // channels)))
    for index in range(len(out)):
        frame = samples[index * channels:(index + 1) * channels]
        out[index] = int(sum(frame) / channels)
    return out


def _resample(samples: array.array, source_rate: int, target_rate: int) -> array.array:
    """Linear interpolation up, box-averaged decimation down.

    The box average is a crude low-pass, but it is what keeps 24 kHz -> 16 kHz
    from folding the top of the band back into the speech. ffmpeg's soxr is
    better and is used whenever it is present; this is the no-dependency path.
    """
    if source_rate == target_rate or not samples:
        return samples
    ratio = source_rate / target_rate
    count = max(1, int(len(samples) / ratio))
    out = array.array("h", bytes(2 * count))
    last = len(samples) - 1

    if ratio > 1.0:                     # downsample: average the source window
        half = ratio / 2.0
        for index in range(count):
            centre = index * ratio
            start = max(0, int(centre - half))
            stop = min(len(samples), int(centre + half) + 1)
            out[index] = int(sum(samples[start:stop]) / (stop - start))
    else:                               # upsample: interpolate between neighbours
        for index in range(count):
            position = index * ratio
            left = int(position)
            right = min(left + 1, last)
            weight = position - left
            out[index] = int(samples[left] * (1.0 - weight) + samples[right] * weight)
    return out


def _via_ffmpeg(data: bytes, fmt: str, sample_rate: Optional[int], channels: int,
                bitrate: str) -> bytes:
    binary = ffmpeg()
    with tempfile.TemporaryDirectory(prefix="afrispeech-audio-") as work:
        source = os.path.join(work, "in.wav")
        target = os.path.join(work, f"out.{fmt}")
        with open(source, "wb") as handle:
            handle.write(data)
        command = [binary, "-y", "-loglevel", "error", "-i", source, "-ac", str(channels)]
        if sample_rate:
            command += ["-ar", str(sample_rate)]
        if fmt not in LOSSLESS:
            command += ["-b:a", bitrate]
        command.append(target)
        result = subprocess.run(command, capture_output=True)
        if result.returncode != 0 or not os.path.exists(target):
            raise AudioError(f"ffmpeg failed: {result.stderr.decode('utf-8', 'replace').strip()}")
        with open(target, "rb") as handle:
            return handle.read()


def convert(data: bytes, fmt: str = "wav", sample_rate: Optional[int] = None,
            channels: int = 1, bitrate: str = "64k") -> Tuple[bytes, int]:
    """Return (audio bytes, effective sample rate).

    `sample_rate=None` keeps whatever the backend produced. WAV in, anything in
    FORMATS out. Untouched input is returned as-is, so the default path copies
    no bytes and spawns no process.
    """
    source_pcm, source_rate, source_channels, bits = parse_wav(data)
    target_rate = sample_rate or source_rate
    if fmt == "wav" and target_rate == source_rate and channels == source_channels:
        return data, source_rate

    if ffmpeg():
        return _via_ffmpeg(data, fmt, sample_rate, channels, bitrate), target_rate

    if fmt != "wav":
        raise AudioError(f"Writing {fmt} needs ffmpeg on PATH.")
    if bits != 16:
        raise AudioError(f"Resampling {bits}-bit audio needs ffmpeg on PATH.")
    if channels != 1:
        raise AudioError("Producing more than one channel needs ffmpeg on PATH.")
    if "resample" not in _WARNED:
        _WARNED.add("resample")
        print(f"  note: resampling {source_rate} -> {target_rate} Hz in Python; "
              f"install ffmpeg for better quality", flush=True)

    samples = array.array("h")
    samples.frombytes(source_pcm[:len(source_pcm) - len(source_pcm) % (2 * source_channels)])
    samples = _resample(_downmix(samples, source_channels), source_rate, target_rate)
    return build_wav(samples.tobytes(), target_rate, 1, 16), target_rate
