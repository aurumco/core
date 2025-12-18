"""Model loading module.

This module handles the loading of quantized models using unsloth or transformers,
handling device mapping and error scenarios.
"""

# Unsloth must be imported first to patch transformers correctly
try:
    import unsloth  # type: ignore # noqa: F401
except ImportError:
    pass

from typing import Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import ModelConfig
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class ModelLoader:
    """Handles loading of the model and tokenizer.

    Attributes:
        config (ModelConfig): The model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the ModelLoader.

        Args:
            config (ModelConfig): Configuration for model loading.
        """
        self.config = config

    def load(self) -> Tuple[Any, Any]:
        """
        Load a causal language model and its tokenizer according to the loader configuration.
        
        Prefers unsloth's FastLanguageModel; if unsloth is not available or fails, falls back to loading the model and tokenizer via Hugging Face transformers.
        
        Returns:
            Tuple[Any, Any]: A tuple (model, tokenizer) containing the loaded model and its tokenizer.
        
        Raises:
            RuntimeError: If both unsloth and transformers loading paths fail.
        """
        try:
            logger.info(
                "Attempting to load model",
                extra={"context": {"model": self.config.model_name}},
            )

            # Try unsloth first as requested
            try:
                from unsloth import FastLanguageModel

                # We load in 16-bit (bfloat16) as requested for "Genetic Engineering"
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.config.model_name,
                    max_seq_length=2048,
                    dtype=torch.bfloat16,
                    load_in_4bit=False,
                )
                logger.info("Loaded model using unsloth (bfloat16)")
                return model, tokenizer
            except ImportError:
                logger.warning("Unsloth not found, falling back to transformers")
            except Exception as e:
                logger.warning(
                    f"Unsloth loading failed: {e}. Falling back to transformers"
                )

            # Fallback to transformers (16-bit)
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.config.device_map,
                trust_remote_code=True,  # nosec B615
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, trust_remote_code=True  # nosec B615
            )  # type: ignore

            logger.info("Loaded model using transformers")
            return model, tokenizer

        except Exception as e:
            logger.error("Failed to load model", exc_info=True)
            raise RuntimeError(f"Model loading failed: {str(e)}") from e