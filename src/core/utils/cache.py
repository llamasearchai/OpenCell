"""
Caching utilities for computationally expensive operations.
"""

import functools
import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

logger = logging.getLogger(__name__)

class CacheConfig:
    """Configuration for cache behavior."""
    
    CACHE_DIR = os.environ.get("OPENCELL_CACHE_DIR", ".cache")
    CACHE_ENABLED = os.environ.get("OPENCELL_CACHE_ENABLED", "1") == "1"
    CACHE_MAX_SIZE_GB = float(os.environ.get("OPENCELL_CACHE_MAX_SIZE_GB", "10"))
    CACHE_EXPIRY_DAYS = int(os.environ.get("OPENCELL_CACHE_EXPIRY_DAYS", "30"))


def cached_computation(func: Callable = None, *, cache_dir: Optional[Union[str, Path]] = None, 
                       expiry_days: Optional[int] = None, enabled: bool = True):
    """
    Decorator for caching expensive computational results.
    
    Args:
        func: The function to decorate
        cache_dir: Directory to store cache files. Defaults to CacheConfig.CACHE_DIR
        expiry_days: Number of days before cache expires. Defaults to CacheConfig.CACHE_EXPIRY_DAYS
        enabled: Whether caching is enabled. Defaults to CacheConfig.CACHE_ENABLED
        
    Returns:
        Decorated function with caching behavior
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Skip caching if disabled globally or for this decorator
            if not (CacheConfig.CACHE_ENABLED and enabled):
                return func(*args, **kwargs)
                
            # Determine cache directory
            cache_directory = Path(cache_dir or CacheConfig.CACHE_DIR)
            cache_directory.mkdir(parents=True, exist_ok=True)
            
            # Calculate a hash of the function name, arguments, and keyword arguments
            cache_key = _generate_cache_key(func.__name__, args, kwargs)
            cache_file = cache_directory / f"{cache_key}.pkl"
            
            # Check if the cache file exists and is valid
            if cache_file.exists() and _is_cache_valid(cache_file, expiry_days or CacheConfig.CACHE_EXPIRY_DAYS):
                try:
                    logger.debug(f"Loading cached result from {cache_file}")
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Error loading cache file {cache_file}: {e}")
            
            # Execute the function and cache the result
            result = func(*args, **kwargs)
            
            try:
                logger.debug(f"Saving result to cache file {cache_file}")
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
            except Exception as e:
                logger.warning(f"Error writing to cache file {cache_file}: {e}")
            
            return result
        return wrapper
        
    # Allow using @cached_computation or @cached_computation(...)
    if func is None:
        return decorator
    return decorator(func)


def _generate_cache_key(func_name: str, args: tuple, kwargs: Dict[str, Any]) -> str:
    """
    Generate a cache key based on function name and arguments.
    
    Args:
        func_name: Name of the function
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        A hash string to use as the cache key
    """
    # Convert args and kwargs to a string representation
    try:
        # For simple arguments
        args_str = json.dumps([str(arg) for arg in args])
        kwargs_str = json.dumps({k: str(v) for k, v in kwargs.items()}, sort_keys=True)
    except TypeError:
        # Fallback for non-serializable arguments
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        
    key_str = f"{func_name}:{args_str}:{kwargs_str}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _is_cache_valid(cache_file: Path, expiry_days: int) -> bool:
    """
    Check if a cache file is still valid based on its age.
    
    Args:
        cache_file: Path to the cache file
        expiry_days: Number of days before cache expires
        
    Returns:
        True if the cache is still valid, False otherwise
    """
    import time
    from datetime import datetime, timedelta
    
    # Get file modification time
    mtime = cache_file.stat().st_mtime
    mtime_dt = datetime.fromtimestamp(mtime)
    
    # Check if the file is older than expiry_days
    return datetime.now() - mtime_dt < timedelta(days=expiry_days)


def clear_cache(cache_dir: Optional[Union[str, Path]] = None, older_than_days: Optional[int] = None):
    """
    Clear the cache directory.
    
    Args:
        cache_dir: Directory containing cache files. Defaults to CacheConfig.CACHE_DIR
        older_than_days: Only clear files older than this many days. If None, clear all.
    """
    import time
    from datetime import datetime, timedelta
    
    cache_directory = Path(cache_dir or CacheConfig.CACHE_DIR)
    if not cache_directory.exists():
        logger.info(f"Cache directory {cache_directory} does not exist.")
        return
        
    count = 0
    size_bytes = 0
    
    for cache_file in cache_directory.glob("*.pkl"):
        # If older_than_days is specified, only delete files older than that
        if older_than_days is not None:
            mtime = cache_file.stat().st_mtime
            mtime_dt = datetime.fromtimestamp(mtime)
            if datetime.now() - mtime_dt < timedelta(days=older_than_days):
                continue
                
        size_bytes += cache_file.stat().st_size
        cache_file.unlink()
        count += 1
        
    logger.info(f"Cleared {count} cache files ({size_bytes / 1024 / 1024:.2f} MB) from {cache_directory}")


def get_cache_stats(cache_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Get statistics about the cache.
    
    Args:
        cache_dir: Directory containing cache files. Defaults to CacheConfig.CACHE_DIR
        
    Returns:
        Dictionary with cache statistics
    """
    cache_directory = Path(cache_dir or CacheConfig.CACHE_DIR)
    
    if not cache_directory.exists():
        return {
            "directory": str(cache_directory),
            "exists": False,
            "file_count": 0,
            "total_size_bytes": 0,
            "total_size_mb": 0,
        }
        
    file_count = 0
    total_size_bytes = 0
    oldest_file_age_days = 0
    newest_file_age_days = float('inf')
    
    import time
    from datetime import datetime, timedelta
    
    now = datetime.now()
    
    for cache_file in cache_directory.glob("*.pkl"):
        file_count += 1
        size = cache_file.stat().st_size
        total_size_bytes += size
        
        mtime = cache_file.stat().st_mtime
        mtime_dt = datetime.fromtimestamp(mtime)
        age_days = (now - mtime_dt).total_seconds() / (24 * 3600)
        
        oldest_file_age_days = max(oldest_file_age_days, age_days)
        newest_file_age_days = min(newest_file_age_days, age_days)
    
    if file_count == 0:
        newest_file_age_days = 0
    
    return {
        "directory": str(cache_directory),
        "exists": True,
        "file_count": file_count,
        "total_size_bytes": total_size_bytes,
        "total_size_mb": total_size_bytes / (1024 * 1024),
        "oldest_file_age_days": oldest_file_age_days,
        "newest_file_age_days": newest_file_age_days,
    } 