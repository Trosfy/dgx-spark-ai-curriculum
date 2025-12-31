"""
Audio Transcription Utilities for DGX Spark

This module provides utilities for transcribing and analyzing audio
using OpenAI's Whisper model on DGX Spark's 128GB unified memory.

Example:
    >>> from scripts.audio_utils import AudioTranscriber
    >>> transcriber = AudioTranscriber()
    >>> result = transcriber.transcribe("meeting.mp3")
    >>> print(result.text)
"""

import gc
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any, Literal
from enum import Enum

import torch
import numpy as np


class WhisperModel(Enum):
    """Available Whisper model sizes."""
    TINY = "tiny"           # 39M params, ~1GB VRAM
    BASE = "base"           # 74M params, ~1GB VRAM
    SMALL = "small"         # 244M params, ~2GB VRAM
    MEDIUM = "medium"       # 769M params, ~5GB VRAM
    LARGE = "large"         # 1550M params, ~10GB VRAM
    LARGE_V2 = "large-v2"   # 1550M params, ~10GB VRAM
    LARGE_V3 = "large-v3"   # 1550M params, ~10GB VRAM


@dataclass
class TranscriptionSegment:
    """A single segment of transcribed audio."""
    id: int
    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str
    tokens: Optional[List[int]] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    language: Optional[str] = None

    @property
    def duration(self) -> float:
        """Duration of this segment in seconds."""
        return self.end - self.start

    def format_timestamp(self, t: float) -> str:
        """Format time as HH:MM:SS.mmm"""
        hours = int(t // 3600)
        minutes = int((t % 3600) // 60)
        seconds = t % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def to_srt(self) -> str:
        """Convert segment to SRT format."""
        start_ts = self.format_timestamp(self.start).replace(".", ",")
        end_ts = self.format_timestamp(self.end).replace(".", ",")
        return f"{self.id}\n{start_ts} --> {end_ts}\n{self.text.strip()}\n"

    def to_vtt(self) -> str:
        """Convert segment to WebVTT format."""
        start_ts = self.format_timestamp(self.start)
        end_ts = self.format_timestamp(self.end)
        return f"{start_ts} --> {end_ts}\n{self.text.strip()}\n"


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    text: str
    segments: List[TranscriptionSegment]
    language: str
    duration: float
    processing_time: float
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_srt(self) -> str:
        """Export as SRT subtitle format."""
        return "\n".join(seg.to_srt() for seg in self.segments)

    def to_vtt(self) -> str:
        """Export as WebVTT subtitle format."""
        header = "WEBVTT\n\n"
        return header + "\n".join(seg.to_vtt() for seg in self.segments)

    def to_json(self) -> str:
        """Export as JSON."""
        data = {
            "text": self.text,
            "language": self.language,
            "duration": self.duration,
            "processing_time": self.processing_time,
            "model": self.model_name,
            "segments": [
                {
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                }
                for s in self.segments
            ],
        }
        return json.dumps(data, indent=2)

    def save(self, path: Union[str, Path], format: str = "json") -> None:
        """
        Save transcription to file.

        Args:
            path: Output file path.
            format: Output format ("json", "srt", "vtt", "txt").
        """
        path = Path(path)

        if format == "json":
            path.write_text(self.to_json())
        elif format == "srt":
            path.write_text(self.to_srt())
        elif format == "vtt":
            path.write_text(self.to_vtt())
        elif format == "txt":
            path.write_text(self.text)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"Saved transcription to {path}")


class AudioTranscriber:
    """
    Transcribe audio using Whisper on DGX Spark.

    This transcriber leverages the 128GB unified memory to run
    the largest Whisper models with ease.

    Attributes:
        model_name: Whisper model to use.
        device: Device for inference.
        compute_type: Computation precision.

    Example:
        >>> transcriber = AudioTranscriber(model="large-v3")
        >>> result = transcriber.transcribe("podcast.mp3")
        >>> print(f"Transcribed {result.duration:.1f}s in {result.processing_time:.1f}s")
    """

    def __init__(
        self,
        model: Union[str, WhisperModel] = WhisperModel.LARGE_V3,
        device: Optional[str] = None,
        compute_type: Literal["float16", "float32", "int8"] = "float16",
    ):
        """
        Initialize the audio transcriber.

        Args:
            model: Whisper model name or enum.
            device: Device for inference (auto-detected if None).
            compute_type: Computation precision.
        """
        if isinstance(model, WhisperModel):
            model = model.value
        self.model_name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type

        self._model = None

    def _ensure_model_loaded(self) -> None:
        """Lazy load the Whisper model."""
        if self._model is not None:
            return

        import whisper

        print(f"Loading Whisper model: {self.model_name}")
        start_time = time.time()

        self._model = whisper.load_model(self.model_name, device=self.device)

        elapsed = time.time() - start_time
        print(f"  Loaded in {elapsed:.1f}s on {self.device}")

        # Report memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"  GPU memory used: {allocated:.1f}GB")

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        task: Literal["transcribe", "translate"] = "transcribe",
        word_timestamps: bool = False,
        initial_prompt: Optional[str] = None,
        temperature: float = 0.0,
        condition_on_previous_text: bool = True,
        verbose: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.).
            language: Source language code (auto-detected if None).
            task: "transcribe" or "translate" (to English).
            word_timestamps: Include word-level timestamps.
            initial_prompt: Optional text to condition the model.
            temperature: Sampling temperature (0 for greedy).
            condition_on_previous_text: Use previous output as context.
            verbose: Print progress.

        Returns:
            TranscriptionResult with text and segments.

        Example:
            >>> result = transcriber.transcribe(
            ...     "interview.mp3",
            ...     language="en",
            ...     word_timestamps=True
            ... )
            >>> print(result.text)
        """
        import whisper

        self._ensure_model_loaded()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Transcribing: {audio_path}")
        start_time = time.time()

        # Load audio
        audio = whisper.load_audio(str(audio_path))
        audio_duration = len(audio) / whisper.audio.SAMPLE_RATE

        if verbose:
            print(f"  Audio duration: {audio_duration:.1f}s")

        # Transcribe
        result = self._model.transcribe(
            audio,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            temperature=temperature,
            condition_on_previous_text=condition_on_previous_text,
            verbose=verbose,
        )

        processing_time = time.time() - start_time

        # Convert segments
        segments = []
        for i, seg in enumerate(result["segments"]):
            segments.append(TranscriptionSegment(
                id=i + 1,
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                tokens=seg.get("tokens"),
                avg_logprob=seg.get("avg_logprob"),
                no_speech_prob=seg.get("no_speech_prob"),
                language=result.get("language"),
            ))

        transcription = TranscriptionResult(
            text=result["text"],
            segments=segments,
            language=result.get("language", "unknown"),
            duration=audio_duration,
            processing_time=processing_time,
            model_name=self.model_name,
            metadata={
                "source": str(audio_path),
                "task": task,
            },
        )

        if verbose:
            ratio = audio_duration / processing_time
            print(f"  Completed in {processing_time:.1f}s ({ratio:.1f}x realtime)")

        return transcription

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        **kwargs,
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files.

        Args:
            audio_paths: List of audio file paths.
            **kwargs: Additional arguments for transcribe().

        Returns:
            List of TranscriptionResult objects.

        Example:
            >>> results = transcriber.transcribe_batch([
            ...     "audio1.mp3",
            ...     "audio2.mp3",
            ...     "audio3.mp3",
            ... ])
        """
        from tqdm import tqdm

        results = []

        for path in tqdm(audio_paths, desc="Transcribing"):
            try:
                result = self.transcribe(path, verbose=False, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error transcribing {path}: {e}")
                results.append(None)

        successful = sum(1 for r in results if r is not None)
        print(f"Transcribed {successful}/{len(audio_paths)} files")

        return results

    def detect_language(
        self,
        audio_path: Union[str, Path],
        top_k: int = 5,
    ) -> Dict[str, float]:
        """
        Detect the language of an audio file.

        Args:
            audio_path: Path to audio file.
            top_k: Number of top languages to return.

        Returns:
            Dict mapping language codes to probabilities.

        Example:
            >>> langs = transcriber.detect_language("mystery.mp3")
            >>> print(f"Most likely: {max(langs, key=langs.get)}")
        """
        import whisper

        self._ensure_model_loaded()

        # Load audio
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)

        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        # Detect language
        _, probs = self._model.detect_language(mel)

        # Get top-k languages
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_probs[:top_k])

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            print("Model unloaded")


def load_audio(
    audio_path: Union[str, Path],
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to target sample rate.

    Args:
        audio_path: Path to audio file.
        target_sr: Target sample rate (16000 for Whisper).

    Returns:
        Tuple of (audio_array, sample_rate).

    Example:
        >>> audio, sr = load_audio("speech.wav")
        >>> print(f"Duration: {len(audio)/sr:.1f}s")
    """
    import librosa

    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio, sr


def split_audio(
    audio_path: Union[str, Path],
    output_dir: Union[str, Path],
    segment_duration: float = 30.0,
    overlap: float = 0.5,
) -> List[Path]:
    """
    Split audio into segments for parallel processing.

    Args:
        audio_path: Path to audio file.
        output_dir: Directory for output segments.
        segment_duration: Duration of each segment in seconds.
        overlap: Overlap between segments in seconds.

    Returns:
        List of paths to segment files.

    Example:
        >>> segments = split_audio("long_podcast.mp3", "segments/", 60.0)
        >>> print(f"Split into {len(segments)} segments")
    """
    import soundfile as sf

    audio, sr = load_audio(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_samples = int(segment_duration * sr)
    overlap_samples = int(overlap * sr)
    hop = segment_samples - overlap_samples

    segments = []
    start = 0
    i = 0

    while start < len(audio):
        end = min(start + segment_samples, len(audio))
        segment = audio[start:end]

        output_path = output_dir / f"segment_{i:04d}.wav"
        sf.write(str(output_path), segment, sr)
        segments.append(output_path)

        start += hop
        i += 1

    print(f"Split audio into {len(segments)} segments")
    return segments


def merge_transcriptions(
    transcriptions: List[TranscriptionResult],
    overlap: float = 0.5,
) -> TranscriptionResult:
    """
    Merge overlapping transcription segments.

    Args:
        transcriptions: List of transcription results.
        overlap: Overlap duration used when splitting.

    Returns:
        Merged transcription result.

    Example:
        >>> results = [transcriber.transcribe(seg) for seg in segments]
        >>> merged = merge_transcriptions(results, overlap=0.5)
    """
    if not transcriptions:
        raise ValueError("No transcriptions to merge")

    if len(transcriptions) == 1:
        return transcriptions[0]

    all_segments = []
    time_offset = 0.0

    for i, trans in enumerate(transcriptions):
        for seg in trans.segments:
            # Adjust timestamps
            adjusted_seg = TranscriptionSegment(
                id=len(all_segments) + 1,
                start=seg.start + time_offset,
                end=seg.end + time_offset,
                text=seg.text,
                language=seg.language,
            )
            all_segments.append(adjusted_seg)

        if i < len(transcriptions) - 1:
            time_offset += trans.duration - overlap

    # Combine text
    full_text = " ".join(seg.text.strip() for seg in all_segments)

    total_duration = sum(t.duration for t in transcriptions) - overlap * (len(transcriptions) - 1)
    total_processing = sum(t.processing_time for t in transcriptions)

    return TranscriptionResult(
        text=full_text,
        segments=all_segments,
        language=transcriptions[0].language,
        duration=total_duration,
        processing_time=total_processing,
        model_name=transcriptions[0].model_name,
        metadata={"merged_from": len(transcriptions)},
    )


def create_audio_qa_prompt(
    transcription: TranscriptionResult,
    question: str,
    include_timestamps: bool = False,
    max_context_length: int = 4000,
) -> str:
    """
    Create a prompt for audio Q&A based on transcription.

    Args:
        transcription: Transcription result.
        question: User's question.
        include_timestamps: Include timestamps in context.
        max_context_length: Maximum context length.

    Returns:
        Formatted prompt for an LLM.

    Example:
        >>> prompt = create_audio_qa_prompt(result, "What was the main topic discussed?")
    """
    if include_timestamps:
        # Build timestamped transcript
        lines = []
        for seg in transcription.segments:
            timestamp = f"[{seg.format_timestamp(seg.start)}]"
            lines.append(f"{timestamp} {seg.text.strip()}")
        transcript = "\n".join(lines)
    else:
        transcript = transcription.text

    # Truncate if needed
    if len(transcript) > max_context_length:
        transcript = transcript[:max_context_length] + "...[truncated]"

    prompt = f"""Audio Transcription:
{transcript}

Based on the audio transcription above, please answer the following question:
{question}

Answer:"""

    return prompt


def get_audio_info(audio_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about an audio file.

    Args:
        audio_path: Path to audio file.

    Returns:
        Dict with audio metadata.

    Example:
        >>> info = get_audio_info("podcast.mp3")
        >>> print(f"Duration: {info['duration']:.1f}s, Channels: {info['channels']}")
    """
    import librosa
    import soundfile as sf

    audio_path = Path(audio_path)

    # Get basic info without loading full audio
    info = sf.info(str(audio_path))

    return {
        "path": str(audio_path),
        "format": info.format,
        "subtype": info.subtype,
        "channels": info.channels,
        "sample_rate": info.samplerate,
        "duration": info.duration,
        "frames": info.frames,
    }


def normalize_audio(
    audio_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    target_db: float = -20.0,
) -> Path:
    """
    Normalize audio volume.

    Args:
        audio_path: Path to input audio.
        output_path: Path for output (overwrites if None).
        target_db: Target loudness in dB.

    Returns:
        Path to normalized audio.

    Example:
        >>> normalized = normalize_audio("quiet.mp3", "loud.mp3", target_db=-16)
    """
    import soundfile as sf

    audio, sr = load_audio(audio_path)

    # Calculate current RMS
    rms = np.sqrt(np.mean(audio**2))
    current_db = 20 * np.log10(rms + 1e-10)

    # Calculate gain
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)

    # Apply gain
    normalized = audio * gain

    # Clip to prevent distortion
    normalized = np.clip(normalized, -1.0, 1.0)

    # Save
    output_path = output_path or audio_path
    sf.write(str(output_path), normalized, sr)

    print(f"Normalized audio from {current_db:.1f}dB to {target_db:.1f}dB")
    return Path(output_path)


if __name__ == "__main__":
    print("Audio Utils - DGX Spark Optimized")
    print("=" * 50)

    print("\nWhisper model sizes and VRAM requirements:")
    for model in WhisperModel:
        print(f"  {model.value}")

    print("\nNote: With 128GB unified memory, you can easily run large-v3!")
