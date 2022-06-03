from setuptools import setup, find_packages
from os import chdir, sep, rename
from os.path import abspath, join, pardir

PROJECT = 'DeepPhysX'
PACKAGE = 'Sofa'
DEPENDENCIES = ['DeepPhysX']

# Check current repository
repository = abspath(join(__file__, pardir)).split(sep)[-1]
if repository != PACKAGE:
    print(f"WARNING: Wrong current repository name, renaming '{repository}' --> '{PACKAGE}'")
    chdir(pardir)
    rename(repository, PACKAGE)
    chdir(PACKAGE)

# Configure packages and subpackages list and dependencies list
packages = [f'{PROJECT}.{PACKAGE}']
packages_dir = {f'{PROJECT}.{PACKAGE}': 'src'}
for sub_package in find_packages(where='src'):
    packages.append(f'{PROJECT}.{PACKAGE}.{sub_package}')
    packages_dir[f'{PROJECT}.{PACKAGE}.{sub_package}'] = join('src', *sub_package.split('.'))

# Installation
setup(name=f'{PROJECT}_{PACKAGE}',
      version='1.0',
      description='A Python framework interfacing AI with numerical simulation.',
      author='Mimesis',
      author_email='robin.enjalbert@inria.fr',
      url='https://github.com/mimesis-inria/DeepPhysX_Sofa',
      packages=packages,
      package_dir=packages_dir,
      install_requires=DEPENDENCIES)
