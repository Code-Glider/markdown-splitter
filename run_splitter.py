from app import process_markdown, SplitterConfig

# Create a configuration (optional, you can skip this and use defaults)
config = SplitterConfig(
    output_extension=".md",  # default extension for output files
    toc_filename="table_of_contents.md"  # name of the table of contents file
)

# Process the markdown file
input_file = "example.md"
output_dir = "split_output"

success = process_markdown(input_file, output_dir, config)

if success:
    print("Markdown file successfully split!")