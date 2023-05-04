from setuptools import setup, find_packages

setup_args = dict(
    name='deepql',
    version='0.1.0',
    description='A package for the physics inverse problem of finding the force giving the trajectory',
    author='Fabian Trigo',
    author_email='fbientrigo@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas'
    ]
)


setup(**setup_args)
