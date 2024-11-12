import pytest
import os
import tempfile
from pathlib import Path
from app import (
    EnhancedMarkdownSplitter,
    SplitterConfig,
    process_markdown,
    MarkdownSplitterError,
    FileNotFoundError,
    OutputDirectoryError,
)

@pytest.fixture
def sample_markdown():
    return """# Header 1
This is content under header 1

## Header 2
This is content under header 2

### Header 3
This is content under header 3

#### Header 4
This is content under header 4

##### Header 5
This is content under header 5
"""

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

def test_splitter_config_defaults():
    config = SplitterConfig()
    assert len(config.headers_to_split_on) == 5
    assert config.output_extension == ".md"
    assert config.encoding == "utf-8"
    assert config.toc_filename == "table_of_contents.md"

def test_splitter_config_custom():
    custom_headers = [("#", "H1"), ("##", "H2")]
    config = SplitterConfig(
        headers_to_split_on=custom_headers,
        output_extension=".markdown",
        encoding="latin-1",
        toc_filename="contents.md"
    )
    assert config.headers_to_split_on == custom_headers
    assert config.output_extension == ".markdown"
    assert config.encoding == "latin-1"
    assert config.toc_filename == "contents.md"

def test_markdown_splitting(sample_markdown):
    splitter = EnhancedMarkdownSplitter()
    sections = splitter.split_text(sample_markdown)
    
    assert len(sections) > 0
    assert all(isinstance(section, dict) for section in sections)
    assert all("header" in section for section in sections)
    assert all("content" in section for section in sections)
    assert all("level" in section for section in sections)
    assert all("metadata" in section for section in sections)

def test_process_markdown_missing_input():
    with pytest.raises(FileNotFoundError):
        process_markdown("nonexistent.md", "output")

def test_process_markdown_invalid_output(temp_dir):
    # Create a file where the output directory should be
    invalid_dir = os.path.join(temp_dir, "invalid_dir")
    with open(invalid_dir, "w") as f:
        f.write("blocking file")
    
    input_file = os.path.join(temp_dir, "input.md")
    with open(input_file, "w") as f:
        f.write("# Test\nContent")
    
    with pytest.raises(OutputDirectoryError):
        process_markdown(input_file, invalid_dir)

def test_successful_processing(temp_dir, sample_markdown):
    # Create input file
    input_file = os.path.join(temp_dir, "input.md")
    with open(input_file, "w") as f:
        f.write(sample_markdown)
    
    # Process markdown
    output_dir = os.path.join(temp_dir, "output")
    result = process_markdown(input_file, output_dir)
    
    assert result is True
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, "table_of_contents.md"))
    
    # Check if files were created
    files = os.listdir(output_dir)
    assert len(files) > 1  # At least TOC and one content file
    assert "table_of_contents.md" in files

def test_custom_config_processing(temp_dir, sample_markdown):
    # Create input file
    input_file = os.path.join(temp_dir, "input.md")
    with open(input_file, "w") as f:
        f.write(sample_markdown)
    
    # Custom configuration
    config = SplitterConfig(
        headers_to_split_on=[("#", "H1"), ("##", "H2")],
        output_extension=".markdown",
        toc_filename="contents.md"
    )
    
    # Process markdown
    output_dir = os.path.join(temp_dir, "output")
    result = process_markdown(input_file, output_dir, config)
    
    assert result is True
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, "contents.md"))
    
    # Check if files use custom extension
    files = [f for f in os.listdir(output_dir) if f.endswith(".markdown")]
    assert len(files) > 0