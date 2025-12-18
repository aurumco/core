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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
        """Loads the model and tokenizer.

        Returns:
            Tuple[Any, Any]: A tuple containing (model, tokenizer).

        Raises:
            RuntimeError: If loading fails.
        """
        try:
            logger.info(
                "Attempting to load model",
                extra={"context": {"model": self.config.model_name}},
            )

            # Try unsloth first as requested
            try:
                from unsloth import FastLanguageModel

                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.config.model_name,
                    max_seq_length=2048,  # Default safe value
                    dtype=None,  # Auto detection
                    load_in_4bit=self.config.quantization_bit == 4,
                )
                logger.info("Loaded model using unsloth")
                return model, tokenizer
            except ImportError:
                logger.warning("Unsloth not found, falling back to transformers")
            except Exception as e:
                logger.warning(
                    f"Unsloth loading failed: {e}. Falling back to transformers"
                )

            # Fallback to transformers
            bnb_config = None
            if self.config.quantization_bit == 4:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )  # type: ignore

            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
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
