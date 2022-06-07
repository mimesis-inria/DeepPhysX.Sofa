from setuptools import setup, find_packages
from os.path import join

PROJECT = 'DeepPhysX'
PACKAGE = 'Sofa'
DEPENDENCIES = ['DeepPhysX']

# Configure packages and subpackages list and dependencies list
packages = [f'{PROJECT}.{PACKAGE}']
packages_dir = {f'{PROJECT}.{PACKAGE}': 'src'}
for sub_package in find_packages(where='src'):
    packages.append(f'{PROJECT}.{PACKAGE}.{sub_package}')
    packages_dir[f'{PROJECT}.{PACKAGE}.{sub_package}'] = join('src', *sub_package.split('.'))

# Extract README.md content
with open('README.md') as f:
    long_description = f.read()

# Installation
setup(name=f'{PROJECT}.{PACKAGE}',
      version='22.06',
      description='A SOFA compatible package for DeepPhysX framework.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Mimesis',
      author_email='robin.enjalbert@inria.fr',
      url='https://github.com/mimesis-inria/DeepPhysX_Sofa',
      packages=packages,
      package_dir=packages_dir,
      install_requires=DEPENDENCIES)
