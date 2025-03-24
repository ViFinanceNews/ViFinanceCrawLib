from setuptools import setup, find_packages

setup(
    name="ViFinanceCrawLib",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "google-generativeai",
        "os",  # Note: 'os' is part of Python's standard library; no need to include it
        "requests",
        "beautifulsoup4",
        "news-please>=1.6.13",
        "pyodbc",
        "httpx",
        "textacy",
        "regex",
        "scikit-learn",
        "detoxify",
        "sentence-transformers",
        "transformers",
        "torch",
        "tf-keras",
        "dotenv",
        "boto3",
        "sagemaker",
        "json",
        "underthesea"
    ],
    description="A Vietnamese Financial Crawling and Analysis Library",
    python_requires='>=3.7',
)