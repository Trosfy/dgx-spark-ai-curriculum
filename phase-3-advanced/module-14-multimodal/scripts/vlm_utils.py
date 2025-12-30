"""
Vision-Language Model Utilities

This module provides utilities for working with Vision-Language Models (VLMs)
like LLaVA and Qwen-VL on DGX Spark.

Example usage:
    from vlm_utils import VLMPipeline

    # Initialize with Qwen-VL
    vlm = VLMPipeline(model_name="qwen")
    vlm.load()

    # Analyze an image
    image = Image.open("photo.jpg")
    response = vlm.ask(image, "What do you see in this image?")
    print(response)

    # Clean up
    vlm.cleanup()
"""

import torch
import gc
import time
import requests
from io import BytesIO
from typing import Optional, List, Dict, Union
from PIL import Image


def clear_gpu_memory() -> None:
    """
    Clear GPU memory cache.

    Should be called before loading new models or after releasing models
    to ensure maximum available memory.

    Example:
        >>> clear_gpu_memory()
        GPU memory cleared!
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("GPU memory cleared!")


def get_memory_usage() -> str:
    """
    Get current GPU memory usage.

    Returns:
        String describing allocated and reserved memory.

    Example:
        >>> print(get_memory_usage())
        Allocated: 8.45GB, Reserved: 10.20GB
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "No GPU available"


def load_image_from_url(url: str, timeout: int = 10) -> Image.Image:
    """
    Load an image from a URL.

    Args:
        url: URL of the image to load.
        timeout: Request timeout in seconds.

    Returns:
        PIL Image in RGB format.

    Raises:
        ValueError: If the URL cannot be fetched or image cannot be loaded.

    Example:
        >>> image = load_image_from_url("https://example.com/photo.jpg")
        >>> print(image.size)
        (800, 600)
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch image from {url}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load image from {url}: {e}")


class VLMPipeline:
    """
    Unified Vision-Language Model Pipeline.

    Supports multiple VLM backends (LLaVA, Qwen-VL) with a consistent interface.
    Optimized for DGX Spark's 128GB unified memory.

    Attributes:
        model_name: Name of the VLM backend ('llava' or 'qwen')
        model: Loaded model instance
        processor: Model processor/tokenizer

    Example:
        >>> vlm = VLMPipeline(model_name="qwen")
        >>> vlm.load()
        >>> image = Image.open("test.jpg")
        >>> answer = vlm.ask(image, "Describe this image.")
        >>> print(answer)
        >>> vlm.cleanup()
    """

    SUPPORTED_MODELS = {
        "llava": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "llava-13b": "llava-hf/llava-v1.6-vicuna-13b-hf",
        "qwen": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen-72b": "Qwen/Qwen2-VL-72B-Instruct",
    }

    def __init__(self, model_name: str = "qwen"):
        """
        Initialize the VLM Pipeline.

        Args:
            model_name: VLM to use. Options: 'llava', 'llava-13b', 'qwen', 'qwen-72b'
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Choose from {list(self.SUPPORTED_MODELS.keys())}")

        self.model_name = model_name
        self.model_id = self.SUPPORTED_MODELS[model_name]
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self, quantize: bool = False) -> None:
        """
        Load the VLM model.

        Args:
            quantize: Whether to use 4-bit quantization (useful for 72B models)
        """
        if self._loaded:
            print(f"{self.model_name} already loaded!")
            return

        clear_gpu_memory()
        print(f"Loading {self.model_name}...")
        print(f"Memory before: {get_memory_usage()}")
        start_time = time.time()

        if self.model_name.startswith("qwen"):
            self._load_qwen(quantize)
        else:
            self._load_llava(quantize)

        load_time = time.time() - start_time
        self._loaded = True
        print(f"\nModel loaded in {load_time:.1f} seconds!")
        print(f"Memory after: {get_memory_usage()}")

    def _load_qwen(self, quantize: bool) -> None:
        """Load Qwen-VL model."""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

        if quantize:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **load_kwargs
        )

    def _load_llava(self, quantize: bool) -> None:
        """Load LLaVA model."""
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

        if quantize:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id,
            **load_kwargs
        )

    def ask(
        self,
        image: Image.Image,
        question: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> str:
        """
        Ask a question about an image.

        Args:
            image: PIL Image to analyze
            question: Question to ask about the image
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Model's response as a string
        """
        if not self._loaded:
            self.load()

        if self.model_name.startswith("qwen"):
            return self._ask_qwen(image, question, max_tokens, temperature)
        else:
            return self._ask_llava(image, question, max_tokens, temperature)

    def _ask_qwen(self, image: Image.Image, question: str, max_tokens: int, temperature: float) -> str:
        """Ask using Qwen-VL."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response

    def _ask_llava(self, image: Image.Image, question: str, max_tokens: int, temperature: float) -> str:
        """Ask using LLaVA."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
            )

        response = self.processor.decode(output[0], skip_special_tokens=True)

        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()

        return response

    def analyze(self, image: Image.Image) -> Dict[str, str]:
        """
        Perform comprehensive analysis of an image.

        Args:
            image: PIL Image to analyze

        Returns:
            Dictionary with description, objects, mood, and text
        """
        analysis = {}

        analysis['description'] = self.ask(image, "Describe this image in 2-3 sentences.")
        analysis['objects'] = self.ask(image, "List the main objects visible in this image.")
        analysis['mood'] = self.ask(image, "What mood or feeling does this image convey?")
        analysis['text'] = self.ask(image, "Is there any text in this image? If yes, what does it say?")

        return analysis

    def cleanup(self) -> None:
        """Release model from memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self._loaded = False
            clear_gpu_memory()
            print(f"{self.model_name} unloaded!")


def batch_analyze_images(
    images: List[Image.Image],
    questions: List[str],
    model_name: str = "qwen"
) -> List[List[str]]:
    """
    Analyze multiple images with multiple questions.

    Args:
        images: List of PIL Images
        questions: List of questions to ask about each image
        model_name: VLM to use

    Returns:
        2D list of responses [image_idx][question_idx]
    """
    vlm = VLMPipeline(model_name)
    vlm.load()

    results = []
    for image in images:
        image_results = []
        for question in questions:
            response = vlm.ask(image, question)
            image_results.append(response)
        results.append(image_results)

    vlm.cleanup()
    return results


if __name__ == "__main__":
    # Demo usage
    print("VLM Utils Demo")
    print("=" * 50)

    # Create a simple test image
    test_image = Image.new('RGB', (224, 224), color='red')

    # Initialize pipeline
    vlm = VLMPipeline(model_name="qwen")
    vlm.load()

    # Ask a question
    response = vlm.ask(test_image, "What color is this image?")
    print(f"Response: {response}")

    # Cleanup
    vlm.cleanup()
