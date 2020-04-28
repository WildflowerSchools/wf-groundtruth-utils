import os
from setuptools import setup, find_packages

BASEDIR = os.path.dirname(os.path.abspath(__file__))
VERSION = open(os.path.join(BASEDIR, 'VERSION')).read().strip()

# Dependencies (format is 'PYPI_PACKAGE_NAME[>=]=VERSION_NUMBER')
BASE_DEPENDENCIES = [
    'boto3>=1.12',
    'botocore>=1.15',
    'click>=7.1.1',
    'opencv-python>=4.2',
    'numpy>=1.18.3',
    'pydantic>=1.5.1'
]

# TEST_DEPENDENCIES = [
# ]
#
DEVELOPMENT_DEPENDENCIES = [
    'autopep8>=1.5.2'
]

# Allow setup.py to be run from any path
os.chdir(os.path.normpath(BASEDIR))

setup(
    name='wf-groundtruth-labeling',
    packages=find_packages(),
    version=VERSION,
    include_package_data=True,
    description='Tools for processing sagemaker/labelbox labeling jobs',
    long_description=open('README.md').read(),
    url='https://github.com/WildflowerSchools/wf-groundtruth-utils',
    author='Benjamin Jaffe-Talberg',
    author_email='ben.talberg@wildflowerschools.org',
    install_requires=BASE_DEPENDENCIES,
    # tests_require=TEST_DEPENDENCIES,
    extras_require={
        'development': DEVELOPMENT_DEPENDENCIES
    },
    entry_points={
        "console_scripts": [
            "groundtruth = groundtruth_utils.cli:cli"
        ]
    },
    keywords=['sagemaker', 'labelbox', 'ground truth', 'S3'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)