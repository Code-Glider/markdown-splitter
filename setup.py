from setuptools import setup, find_packages

setup(
    name="markdown-file-splitter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.0.200",
        "python-dotenv>=0.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "black>=23.7.0",
            "flake8>=6.1.0",
        ],
    },
    python_requires=">=3.7",
    author="OpenHands",
    author_email="openhands@all-hands.dev",
    description="A Python tool that intelligently splits Markdown files based on header levels",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/openhands/markdown-file-splitter",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)