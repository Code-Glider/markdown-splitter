

```markdown
# Markdown File Splitter

A Python tool that intelligently splits Markdown files into separate files based on header levels (# and ##) while maintaining a comprehensive table of contents that includes all header levels (# through #####). Built with LangChain for enhanced metadata handling.

## Features

- **Smart Splitting**: Splits markdown files at # and ## headers into separate files
- **Comprehensive TOC**: Generates table of contents including all header levels (# to #####)
- **Clean Filenames**: Creates URL-friendly filenames from headers using underscores
- **Metadata Preservation**: Uses LangChain for enhanced metadata handling
- **Hierarchy Maintenance**: Preserves document structure and header relationships
- **Flexible Output**: Customizable output directory for split files

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/markdown-file-splitter.git

# Navigate to the directory
cd 
# Install required packages
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from markdown_splitter import process_markdown

# Process a markdown file
input_file = "your_markdown_file.md"
output_directory = "split_markdown"
process_markdown(input_file, output_directory)
```

### Output Structure

```
output_directory/
├── table_of_contents.md
├── introduction.md
├── getting_started.md
└── advanced_features.md
```

## Requirements

- Python 3.7+
- LangChain library
- Operating System: Windows, macOS, or Linux

## How It Works

1. **File Reading**: Reads the input markdown file
2. **Header Processing**: Identifies all header levels (# through #####)
3. **Content Splitting**: Splits content at # and ## headers
4. **Metadata Extraction**: Uses LangChain to extract and preserve metadata
5. **File Generation**: Creates separate files for each section
6. **TOC Creation**: Generates a comprehensive table of contents

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Your Name - [your@email.com](mailto:your@email.com)

## Acknowledgments

- LangChain for metadata handling capabilities
- Markdown community for inspiration and best practices

## Built With

- [Python](https://www.python.org/) - Primary programming language
- [LangChain](https://python.langchain.com/) - For metadata extraction and handling
```

This README follows best practices by:
1. Starting with a clear project title and description
2. Including all essential sections (Features, Installation, Usage, etc.)
3. Using proper markdown formatting and hierarchy
4. Providing code examples and directory structure
5. Including contribution guidelines and license information
6. Adding contact information and acknowledgments
7. Listing technologies used

Remember to:
- Update the GitHub repository URL
- Add your contact information
- Customize the license section
- Add any specific requirements or dependencies
- Include any additional sections relevant to your project


