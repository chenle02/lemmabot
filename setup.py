from setuptools import setup

setup(
    name='chatpdf',
    version='0.1.0',
    description='Local Research Assistant CLI for PDF documents',
    license='MIT',
    py_modules=['chatpdf'],
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
            'chatpdf=chatpdf:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)