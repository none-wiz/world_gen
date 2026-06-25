import gc
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoryManager")

def clear_gpu_memory():
    """Aggressively clears GPU VRAM and System RAM cache."""
    logger.info("Clearing memory/cache...")
    
    # Run garbage collection
    gc.collect()
    
    # Clear CUDA cache if GPU is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Log VRAM Usage
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        logger.info(f"VRAM Allocated: {allocated:.2f} GB | VRAM Reserved: {reserved:.2f} GB")

class MemoryGuard:
    """Context manager to ensure cleanup after a block of code execution."""
    def __enter__(self):
        clear_gpu_memory()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_gpu_memory()
        if exc_type:
            logger.error(f"Error occurred during execution: {exc_val}")
        return False  # Do not suppress exceptions
