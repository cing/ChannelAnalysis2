from distutils.core import setup

setup(
    name='ChannelAnalysis2',
    version='0.1.0',
    author='Christopher Ing',
    author_email='ing.chris@gmail.com',
    packages=['ChannelAnalysis2'],
    url='https://github.com/cing/ChannelAnalysis2/',
    license='LICENSE',
    description='An analysis pipeline for studying ion channels.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
    ],
)
