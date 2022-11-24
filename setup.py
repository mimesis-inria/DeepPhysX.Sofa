from setuptools import setup, find_packages
from os.path import join

PROJECT = 'DeepPhysX'
PACKAGE = 'Sofa'

packages = [f'{PROJECT}.{PACKAGE}']
packages_dir = {f'{PROJECT}.{PACKAGE}': 'src'}

# Configure packages list and directories
for sub_package in find_packages(where='src'):
    packages.append(f'{PROJECT}.{PACKAGE}.{sub_package}')
    packages_dir[f'{PROJECT}.{PACKAGE}.{sub_package}'] = join('src', *sub_package.split('.'))

# Add examples as subpackages
packages.append(f'{PROJECT}.examples.{PACKAGE}')
packages_dir[f'{PROJECT}.examples.{PACKAGE}'] = 'examples'
for example_dir in find_packages(where='examples'):
    packages.append(f'{PROJECT}.examples.{PACKAGE}.{example_dir}')

# Extract README.md content
with open('README.md') as f:
    long_description = f.read()

# Installation
setup(name=f'{PROJECT}.{PACKAGE}',
      version='22.12',
      description='A SOFA compatibility layer for DeepPhysX framework.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Mimesis',
      author_email='robin.enjalbert@inria.fr',
      url='https://github.com/mimesis-inria/DeepPhysX.Sofa',
      packages=packages,
      package_dir=packages_dir,
      install_requires=['DeepPhysX'])
