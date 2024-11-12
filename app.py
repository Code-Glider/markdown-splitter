from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
import os
import re
import sys
import time
import logging
import psutil
from typing import List, Dict, Any, Optional, Tuple, Callable, TypeVar
from dataclasses import dataclass
import pathlib
from datetime import datetime
import shutil
from functools import wraps
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('markdown_splitter.log')
    ]
)
logger = logging.getLogger(__name__)

class MarkdownSplitterError(Exception):
    """Base exception for markdown splitter errors"""
    pass

class FileNotFoundError(MarkdownSplitterError):
    """Raised when input file is not found"""
    pass

class OutputDirectoryError(MarkdownSplitterError):
    """Raised when there are issues with the output directory"""
    pass

class MemoryError(MarkdownSplitterError):
    """Raised when there's insufficient memory"""
    pass

class IOTimeoutError(MarkdownSplitterError):
    """Raised when I/O operations timeout"""
    pass

class ProcessingError(MarkdownSplitterError):
    """Raised when there's an error processing the markdown"""
    pass

# Type variable for generic function type hints
T = TypeVar('T')

def check_memory_usage(threshold_percent: float = 90.0) -> None:
    """
    Check if memory usage is below threshold
    
    Args:
        threshold_percent: Maximum allowed memory usage percentage
    
    Raises:
        MemoryError: If memory usage exceeds threshold
    """
    memory = psutil.virtual_memory()
    if memory.percent >= threshold_percent:
        msg = f"Memory usage too high: {memory.percent}% >= {threshold_percent}%"
        logger.error(msg)
        raise MemoryError(msg)

def with_retry(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Retry decorator with exponential backoff
    
    Args:
        retries: Number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_delay = delay
            last_exception = None
            
            for attempt in range(retries + 1):
                try:
                    # Check memory before each attempt
                    check_memory_usage()
                    
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt}/{retries} for {func.__name__}")
                    
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    if attempt == retries:
                        logger.error(f"Failed all {retries} retries for {func.__name__}: {str(e)}")
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    logger.warning(f"Waiting {current_delay:.1f}s before next retry")
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception  # Should never reach here
        
        return wrapper
    return decorator

def with_timeout(timeout_seconds: float) -> Callable:
    """
    Timeout decorator using ThreadPoolExecutor
    
    Args:
        timeout_seconds: Maximum execution time in seconds
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except TimeoutError:
                    logger.error(f"Function {func.__name__} timed out after {timeout_seconds}s")
                    raise IOTimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        return wrapper
    return decorator

@dataclass
class SplitterConfig:
    """Configuration for markdown splitter"""
    headers_to_split_on: List[tuple] = None
    output_extension: str = ".md"
    encoding: str = "utf-8"
    toc_filename: str = "table_of_contents.md"
    
    def __post_init__(self):
        if self.headers_to_split_on is None:
            self.headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
                ("#####", "Header 5")
            ]

class EnhancedMarkdownSplitter:
    def __init__(self, config: Optional[SplitterConfig] = None):
        self.config = config or SplitterConfig()
        # LangChain's splitter for metadata extraction
        self.langchain_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.config.headers_to_split_on
        )

    def split_text(self, text: str) -> List[Dict[str, Any]]:
        # First, split using LangChain for metadata
        langchain_splits = self.langchain_splitter.split_text(text)
        
        # Custom splitting logic for file creation
        parts = []
        lines = text.split('\n')
        current_h1 = None
        current_h1_content = []
        current_h2 = None
        current_h2_content = []

        for line in lines:
            if line.startswith('# '):
                if current_h2:
                    parts.append({
                        'header': current_h2,
                        'content': '\n'.join(current_h2_content),
                        'level': 2
                    })
                    current_h2 = None
                    current_h2_content = []
                
                if current_h1:
                    parts.append({
                        'header': current_h1,
                        'content': '\n'.join(current_h1_content),
                        'level': 1
                    })
                
                current_h1 = line
                current_h1_content = []
                
            elif line.startswith('## '):
                if current_h2:
                    parts.append({
                        'header': current_h2,
                        'content': '\n'.join(current_h2_content),
                        'level': 2
                    })
                
                current_h2 = line
                current_h2_content = []
                
            else:
                if current_h2:
                    current_h2_content.append(line)
                elif current_h1:
                    current_h1_content.append(line)

        # Save last sections
        if current_h2:
            parts.append({
                'header': current_h2,
                'content': '\n'.join(current_h2_content),
                'level': 2
            })
        elif current_h1:
            parts.append({
                'header': current_h1,
                'content': '\n'.join(current_h1_content),
                'level': 1
            })

        # Combine LangChain metadata with our splits
        return self.merge_splits(parts, langchain_splits)

    def merge_splits(self, parts, langchain_splits):
        """Merge custom splits with LangChain metadata"""
        merged_parts = []
        for part in parts:
            metadata = self.find_matching_metadata(part, langchain_splits)
            merged_parts.append({
                **part,
                'metadata': metadata
            })
        return merged_parts

    def find_matching_metadata(self, part, langchain_splits):
        """Find matching metadata from LangChain splits"""
        header_text = part['header'].lstrip('#').strip()
        for split in langchain_splits:
            if header_text in str(split.metadata):
                return split.metadata
        return {}

def create_output_subfolder(base_output_dir: str, input_filename: str) -> str:
    """Create a timestamped subfolder for output files"""
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Get input filename without extension
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    # Create subfolder name
    subfolder_name = f"{base_name}_{timestamp}"
    # Create full path
    output_subfolder = os.path.join(base_output_dir, subfolder_name)
    
    try:
        os.makedirs(output_subfolder, exist_ok=True)
    except Exception as e:
        raise OutputDirectoryError(f"Failed to create output subfolder: {str(e)}")
    
    return output_subfolder

def validate_paths(input_file: str, output_dir: str) -> None:
    """Validate input file and output directory paths"""
    input_path = pathlib.Path(input_file)
    output_path = pathlib.Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input path is not a file: {input_file}")
    
    # Check if output directory exists or can be created
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise OutputDirectoryError(f"Permission denied when creating directory: {output_dir}")
    except Exception as e:
        raise OutputDirectoryError(f"Error creating output directory: {str(e)}")

    # Verify output directory is writable
    if not os.access(output_dir, os.W_OK):
        raise OutputDirectoryError(f"Output directory is not writable: {output_dir}")

def find_markdown_files(input_dir: str) -> List[str]:
    """Find all markdown files in the input directory"""
    markdown_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.md', '.markdown')):
            markdown_files.append(os.path.join(input_dir, file))
    return markdown_files

@with_retry(retries=3, delay=1.0, backoff=2.0, 
           exceptions=(IOError, MemoryError, ProcessingError))
@with_timeout(timeout_seconds=30.0)
def read_markdown_file(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Read markdown file with retries and timeout
    
    Args:
        file_path: Path to the markdown file
        encoding: File encoding
    
    Returns:
        str: Content of the file
        
    Raises:
        IOTimeoutError: If reading takes too long
        ProcessingError: If file cannot be read
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
            logger.info(f"Successfully read file: {file_path}")
            return content
    except UnicodeDecodeError as e:
        raise ProcessingError(f"Failed to decode file {file_path} with encoding {encoding}: {str(e)}")
    except Exception as e:
        raise ProcessingError(f"Error reading file {file_path}: {str(e)}")

@with_retry(retries=3, delay=1.0, backoff=2.0,
           exceptions=(IOError, MemoryError, ProcessingError))
@with_timeout(timeout_seconds=30.0)
def write_markdown_file(content: str, file_path: str, encoding: str = 'utf-8') -> None:
    """
    Write markdown file with retries and timeout
    
    Args:
        content: Content to write
        file_path: Path where to write the file
        encoding: File encoding
        
    Raises:
        IOTimeoutError: If writing takes too long
        ProcessingError: If file cannot be written
    """
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
            logger.info(f"Successfully wrote file: {file_path}")
    except Exception as e:
        raise ProcessingError(f"Error writing file {file_path}: {str(e)}")

def save_markdown_sections(input_file: str, output_dir: str, config: Optional[SplitterConfig] = None):
    """Save markdown sections to separate files with enhanced error handling"""
    config = config or SplitterConfig()
    
    # Validate paths
    validate_paths(input_file, output_dir)
    
    # Read input file with retries and timeout
    markdown_content = read_markdown_file(input_file, config.encoding)
    
    # Initialize and use enhanced splitter
    try:
        splitter = EnhancedMarkdownSplitter(config)
        sections = splitter.split_text(markdown_content)
    except Exception as e:
        raise ProcessingError(f"Error splitting markdown content: {str(e)}")
    
    # Create TOC
    toc = ["# Table of Contents\n"]
    current_h1 = None
    
    # Process sections and create files
    try:
        for section in sections:
            # Clean header for filename
            clean_header = section['header'].lstrip('#').strip()
            # Replace spaces with underscores and remove non-alphanumeric characters
            filename = re.sub(r'[\s]+', '_', clean_header)
            filename = re.sub(r'[^\w_-]', '', filename).strip('_').lower() + config.output_extension
            
            # Write section content
            output_path = os.path.join(output_dir, filename)
            full_content = f"{section['header']}\n{section['content']}"
            write_markdown_file(full_content, output_path, config.encoding)
            
            # Update TOC
            if section['level'] == 1:
                current_h1 = clean_header
                toc.append(f"- [{clean_header}]({filename})")
            elif section['level'] == 2:
                toc.append(f"  - [{clean_header}]({filename})")
            
            # Add lower-level headers to TOC
            if 'metadata' in section:
                for header_level in ['Header 3', 'Header 4', 'Header 5']:
                    if header_level in section['metadata']:
                        indent = "  " * (int(header_level[-1]) - 1)
                        header_text = section['metadata'][header_level]
                        toc.append(f"{indent}- {header_text}")
        
        # Write TOC file
        toc_path = os.path.join(output_dir, config.toc_filename)
        write_markdown_file('\n'.join(toc), toc_path, config.encoding)
        
    except Exception as e:
        # Clean up on failure
        logger.error(f"Error processing sections: {str(e)}")
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                logger.info(f"Cleaned up failed output directory: {output_dir}")
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up directory {output_dir}: {str(cleanup_error)}")
        raise ProcessingError(f"Failed to process markdown sections: {str(e)}")

def process_markdown(input_file: str, output_dir: str, config: Optional[SplitterConfig] = None) -> Tuple[bool, str]:
    """
    Process a single markdown file
    
    Args:
        input_file: Path to the input markdown file
        output_dir: Base directory where output will be saved
        config: Optional configuration object
        
    Returns:
        Tuple[bool, str]: (success status, path to output subfolder)
        
    Raises:
        MarkdownSplitterError: Base exception for all splitter errors
        FileNotFoundError: When input file is not found
        OutputDirectoryError: When there are issues with the output directory
    """
    # Create timestamped subfolder for this file
    output_subfolder = create_output_subfolder(output_dir, input_file)
    
    try:
        save_markdown_sections(input_file, output_subfolder, config)
        print(f"Successfully processed {input_file}")
        print(f"Output files saved to {output_subfolder}")
        print(f"Table of contents created as '{config.toc_filename if config else 'table_of_contents.md'}'")
        return True, output_subfolder
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        # Clean up failed output directory
        if os.path.exists(output_subfolder):
            shutil.rmtree(output_subfolder)
        return False, ""

def process_input_directory(input_dir: str = "input", output_dir: str = "output", config: Optional[SplitterConfig] = None) -> List[Tuple[str, str, bool]]:
    """
    Process all markdown files in the input directory
    
    Args:
        input_dir: Directory containing input markdown files
        output_dir: Directory where output folders will be created
        config: Optional configuration object
    
    Returns:
        List[Tuple[str, str, bool]]: List of (input file, output folder, success status)
    """
    # Validate directories
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input path is not a directory: {input_dir}")
    
    # Find all markdown files
    markdown_files = find_markdown_files(input_dir)
    if not markdown_files:
        print(f"No markdown files found in {input_dir}")
        return []
    
    results = []
    for input_file in markdown_files:
        success, output_subfolder = process_markdown(input_file, output_dir, config)
        results.append((input_file, output_subfolder, success))
    
    # Print summary
    print("\nProcessing Summary:")
    print("-" * 50)
    for input_file, output_subfolder, success in results:
        status = "✓ Success" if success else "✗ Failed"
        print(f"{status}: {os.path.basename(input_file)} -> {os.path.basename(output_subfolder)}")
    
    return results

if __name__ == "__main__":
    # Example usage with custom configuration
    config = SplitterConfig(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ],
        output_extension=".markdown",
        toc_filename="contents.md"
    )
    
    try:
        results = process_input_directory("input", "output", config)
        successful = sum(1 for _, _, success in results if success)
        total = len(results)
        print(f"\nProcessed {successful}/{total} files successfully")
    except Exception as e:
        print(f"Error: {str(e)}")