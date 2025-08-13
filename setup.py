from setuptools import setup, find_packages

setup(
    name="namedropper",
    version="0.1.0",
    description="Biomedical ontology harmonization system with AI-powered mapping",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/namedropper",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "psycopg2-binary>=2.9.0",
        "gradio>=4.0.0",
        "langchain-ollama>=0.1.0",
        "tqdm>=4.64.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.21.0",
        "rdflib>=6.0.0",
        "requests>=2.28.0",
    ],
    entry_points={
        "console_scripts": [
            "namedropper=cli.harmonize_metadata:main",
            "namedropper-manage=cli.ontology_manager:main",
            "namedropper-web=web.app:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
