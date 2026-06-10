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
    # Dual import surface: consumers use both `rignak.<mod>` and `rignak.src.<mod>`.
    # Map both package names (plus the custom_requests subpackage under each)
    # to the same src/ tree so either form resolves after `pip install`.
    # `rignak.src.init` resolves to src/init.py.
    packages=[
        'rignak',
        'rignak.custom_requests',
        'rignak.src',
        'rignak.src.custom_requests',
    ],
    package_dir={
        "rignak": "src",
        "rignak.custom_requests": "src/custom_requests",
        "rignak.src": "src",
        "rignak.src.custom_requests": "src/custom_requests",
    },
    install_requires=[
        "requests",
        "beautifulsoup4",
        "Pillow",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        'pytest',
        "lxml",
        'basemap'
        # stem is optional, see extras_require
    ],
    extras_require={
        'tor': ['stem'],
        'display': ['matplotlib', 'seaborn', 'numpy']  # Seaborn also often needs pandas
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
