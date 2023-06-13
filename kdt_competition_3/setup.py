from setuptools import find_packages, setup


setup(
    name='kdt_competition_3',
    version='0.1',
    packages=find_packages(where='.'),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'torchvision',
        'torchmetrics',
        'opencv-python'
    ]
)