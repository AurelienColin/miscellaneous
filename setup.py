import os
from setuptools import setup, find_packages

setup(
    name="rignak",
    version="0.1.0",
    description="A collection of Python utilities.",
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type="text/markdown",
    author="Rignak",
    author_email="",  # Add email if available or leave empty
    url="",  # Add project URL if available
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests",
        "beautifulsoup4",
        "Pillow",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        # stem is optional, see extras_require
    ],
    extras_require={
        'tor': ['stem'],
        'display': ['matplotlib', 'seaborn', 'numpy'] # Seaborn also often needs pandas
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Assuming MIT, adjust if different
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
