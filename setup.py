from setuptools import setup
import pathlib

# Read the long description from the README file
HERE = pathlib.Path(__file__).parent
long_description = (HERE / "Readme.md").read_text(encoding="utf-8")

setup(
    name='lemmabot',
    version='0.2.0',
    description='Local Research Assistant CLI for PDF documents',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    py_modules=['lemmabot'],
    install_requires=[
        'openai>=0.27.0',
        'PyPDF2>=3.0.0',
        'numpy>=1.20.0',
        'python-dotenv>=0.21.0',
        'faiss-cpu>=1.7.1',
        'tiktoken>=0.3.0',
        'requests>=2.28.0',
        'lxml>=4.9.0',
        'unstructured>=0.7.3'
    ],
    entry_points={
        'console_scripts': [
            'lemmabot=lemmabot:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)