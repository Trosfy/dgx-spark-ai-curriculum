"""
Audio Transcription Utilities

This module provides utilities for audio transcription using Whisper
and audio Q&A using LLMs on DGX Spark.

Example usage:
    from audio_utils import AudioTranscriber

    # Initialize
    transcriber = AudioTranscriber()
    transcriber.load()

    # Transcribe audio
    audio, sr = load_audio("recording.wav")
    result = transcriber.transcribe(audio)
    print(result['text'])

    # Clean up
    transcriber.cleanup()
"""

import torch
import gc
import time
import numpy as np
from typing import Optional, List, Dict, Tuple


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_usage() -> str:
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        return f"Allocated: {allocated:.2f}GB"
    return "No GPU"


def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio from a file.

    Args:
        file_path: Path to audio file (wav, mp3, flac, etc.)
        target_sr: Target sample rate (Whisper expects 16000)

    Returns:
        Tuple of (audio_data, sample_rate)

    Example:
        >>> audio, sr = load_audio("speech.wav")
        >>> print(f"Duration: {len(audio) / sr:.2f}s")
    """
    import librosa

    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr


def save_audio(file_path: str, audio: np.ndarray, sample_rate: int = 16000) -> None:
    """
    Save audio to a file.

    Args:
        file_path: Output file path
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz

    Example:
        >>> save_audio("output.wav", audio, 16000)
    """
    import soundfile as sf

    sf.write(file_path, audio, sample_rate)


def generate_sine_wave(
    frequency: float,
    duration: float,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Generate a sine wave tone.

    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Audio array as float32

    Example:
        >>> tone = generate_sine_wave(440, 1.0)  # A4 for 1 second
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * frequency * t).astype(np.float32)


class AudioTranscriber:
    """
    Audio Transcription Pipeline.

    Uses Whisper for speech-to-text transcription with support
    for multiple languages and timestamps.

    Attributes:
        model: Loaded Whisper model
        processor: Whisper processor
        model_size: Size of loaded model

    Example:
        >>> transcriber = AudioTranscriber(model_size="large-v3")
        >>> transcriber.load()
        >>> result = transcriber.transcribe(audio)
        >>> print(result['text'])
    """

    MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

    def __init__(self, model_size: str = "large-v3"):
        """
        Initialize Audio Transcriber.

        Args:
            model_size: Whisper model size
        """
        if model_size not in self.MODEL_SIZES:
            raise ValueError(f"Invalid model size. Choose from {self.MODEL_SIZES}")

        self.model_size = model_size
        self.model = None
        self.processor = None
        self._loaded = False
        self._use_hf = True  # Use Hugging Face version by default

    def load(self, use_hf: bool = True) -> None:
        """
        Load the Whisper model.

        Args:
            use_hf: If True, use Hugging Face transformers version
        """
        if self._loaded:
            return

        clear_gpu_memory()
        self._use_hf = use_hf

        print(f"Loading Whisper {self.model_size}...")
        start_time = time.time()

        if use_hf:
            self._load_hf()
        else:
            self._load_openai()

        load_time = time.time() - start_time
        self._loaded = True
        print(f"Loaded in {load_time:.1f}s")
        print(f"Memory: {get_memory_usage()}")

    def _load_hf(self) -> None:
        """Load Hugging Face Whisper."""
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        model_id = f"openai/whisper-{self.model_size}"

        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16  # Optimized for Blackwell
        ).to("cuda")

    def _load_openai(self) -> None:
        """Load OpenAI Whisper."""
        import whisper

        self.model = whisper.load_model(self.model_size)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        return_timestamps: bool = False
    ) -> Dict:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of audio
            language: Language code (e.g., 'en', 'es') or None for auto-detect
            return_timestamps: Whether to return word timestamps

        Returns:
            Dictionary with transcription results

        Example:
            >>> result = transcriber.transcribe(audio)
            >>> print(result['text'])
            >>> print(f"Duration: {result['duration']:.2f}s")
        """
        if not self._loaded:
            self.load()

        # Resample if needed
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        # Ensure correct format
        audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / np.abs(audio).max()

        start_time = time.time()

        if self._use_hf:
            result = self._transcribe_hf(audio, language, return_timestamps)
        else:
            result = self._transcribe_openai(audio, language)

        transcribe_time = time.time() - start_time

        result['duration'] = len(audio) / 16000
        result['transcribe_time'] = transcribe_time
        result['rtf'] = transcribe_time / result['duration']

        return result

    def _transcribe_hf(
        self,
        audio: np.ndarray,
        language: Optional[str],
        return_timestamps: bool
    ) -> Dict:
        """Transcribe using Hugging Face."""
        input_features = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.model.device, dtype=torch.float16)

        forced_decoder_ids = None
        if language:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language,
                task="transcribe"
            )

        with torch.inference_mode():
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                return_timestamps=return_timestamps,
                max_new_tokens=448
            )

        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return {'text': transcription}

    def _transcribe_openai(
        self,
        audio: np.ndarray,
        language: Optional[str]
    ) -> Dict:
        """Transcribe using OpenAI Whisper."""
        result = self.model.transcribe(
            audio,
            language=language,
            fp16=torch.cuda.is_available()
        )

        return {
            'text': result['text'],
            'language': result['language'],
            'segments': result['segments']
        }

    def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None
    ) -> Dict:
        """
        Transcribe an audio file.

        Args:
            file_path: Path to audio file
            language: Language code or None for auto-detect

        Returns:
            Transcription results
        """
        audio, sr = load_audio(file_path)
        return self.transcribe(audio, sr, language)

    def cleanup(self) -> None:
        """Release resources."""
        if self.model is not None:
            del self.model
            if self.processor is not None:
                del self.processor
            self.model = None
            self.processor = None
            self._loaded = False
            clear_gpu_memory()


class AudioQA:
    """
    Audio Question-Answering Pipeline.

    Combines Whisper transcription with LLM for answering
    questions about audio content.

    Example:
        >>> qa = AudioQA()
        >>> qa.load()
        >>> qa.add_transcript("meeting", "We discussed the Q4 budget...")
        >>> answer = qa.ask("What was discussed?")
    """

    def __init__(self):
        """Initialize Audio QA."""
        self.transcriber = None
        self.llm = None
        self.tokenizer = None
        self.transcripts: Dict[str, Dict] = {}
        self._loaded = False

    def load(self, whisper_size: str = "large-v3") -> None:
        """
        Load models.

        Args:
            whisper_size: Whisper model size
        """
        if self._loaded:
            return

        self.transcriber = AudioTranscriber(whisper_size)
        self.transcriber.load()
        self._loaded = True

    def add_audio(
        self,
        audio_id: str,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> str:
        """
        Transcribe and add audio to the collection.

        Args:
            audio_id: Unique identifier
            audio: Audio data
            sample_rate: Sample rate

        Returns:
            Transcription text
        """
        if not self._loaded:
            self.load()

        result = self.transcriber.transcribe(audio, sample_rate)

        self.transcripts[audio_id] = {
            'text': result['text'],
            'duration': result['duration'],
            'timestamp': time.time()
        }

        return result['text']

    def add_transcript(self, audio_id: str, text: str, duration: float = 0.0) -> None:
        """
        Add a pre-transcribed text.

        Args:
            audio_id: Unique identifier
            text: Transcript text
            duration: Audio duration in seconds
        """
        self.transcripts[audio_id] = {
            'text': text,
            'duration': duration,
            'timestamp': time.time()
        }

    def _load_llm(self) -> None:
        """Load LLM for Q&A."""
        if self.llm is not None:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("Loading LLM for Q&A...")
        model_id = "Qwen/Qwen2.5-7B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("LLM loaded!")

    def ask(self, question: str, audio_id: Optional[str] = None) -> str:
        """
        Ask a question about transcribed audio.

        Args:
            question: Question to ask
            audio_id: Specific audio ID (None = use all)

        Returns:
            Answer string
        """
        if not self.transcripts:
            return "No transcripts available."

        self._load_llm()

        if audio_id:
            if audio_id not in self.transcripts:
                return f"Audio '{audio_id}' not found."
            context = f"Transcript: {self.transcripts[audio_id]['text']}"
        else:
            parts = [f"{aid}: {data['text']}" for aid, data in self.transcripts.items()]
            context = "Transcripts:\n" + "\n\n".join(parts)

        messages = [
            {"role": "system", "content": "Answer questions based on audio transcripts."},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.llm.device)

        with torch.inference_mode():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response

    def summarize(self, audio_id: str) -> str:
        """
        Summarize a transcript.

        Args:
            audio_id: Audio to summarize

        Returns:
            Summary text
        """
        return self.ask("Summarize the main points.", audio_id)

    def list_transcripts(self) -> List[Dict]:
        """Get list of transcripts."""
        return [
            {'id': aid, 'duration': data['duration'], 'preview': data['text'][:100]}
            for aid, data in self.transcripts.items()
        ]

    def cleanup(self) -> None:
        """Release resources."""
        if self.transcriber is not None:
            self.transcriber.cleanup()
        if self.llm is not None:
            del self.llm
            del self.tokenizer
            self.llm = None
            self.tokenizer = None
        clear_gpu_memory()


if __name__ == "__main__":
    print("Audio Utils Demo")
    print("=" * 50)

    # Generate test audio (simple tone)
    audio = generate_sine_wave(440, 2.0)
    print(f"Generated {len(audio) / 16000:.2f}s of audio")

    # Initialize transcriber
    transcriber = AudioTranscriber(model_size="large-v3")
    transcriber.load()

    # Transcribe (will be empty/noise since it's just a tone)
    result = transcriber.transcribe(audio)
    print(f"Transcription: '{result['text']}'")
    print(f"Processing time: {result['transcribe_time']:.2f}s")
    print(f"Real-time factor: {result['rtf']:.2f}x")

    transcriber.cleanup()
