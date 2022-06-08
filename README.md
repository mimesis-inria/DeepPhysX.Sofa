# DeepPhysX.Sofa

This python package is part of the [DeepPhysX](https://github.com/mimesis-inria/DeepPhysX) project.
It contains adaptations of some Core components that are compatible with the [SOFA](https://www.sofa-framework.org/) 
framework.

### Quick install

The package requires [DeepPhysX](https://github.com/mimesis-inria/DeepPhysX) to be installed.
Furthermore, it will require [SOFA bindings](https://sofapython3.readthedocs.io/en/latest/) to be used.

The easiest way to install is using `pip`, but there are a several way to install and configure a **DeepPhysX**
environment (refer to the [**documentation**](https://deepphysx.readthedocs.io) for further instructions).

```bash
$ pip install DeepPhysX.Sofa
```

If cloning sources, clone it in the same repository as other `DeepPhysX` packages.
It must be cloned in a directory with the corresponding name as shown below:

``` bash
$ mkdir DeepPhysX
$ cd DeepPhysX
$ git clone https://github.com/mimesis-inria/DeepPhysX.git Core             # Clone default package
$ git clone https://github.com/mimesis-inria/DeepPhysX.Sofa.git Sofa        # Clone simulation package
$ ls
Core Sofa
```
