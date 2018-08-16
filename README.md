======
b2plot
======




Description
===========

Style and plotting tools for matplotlib.

Installation
============

```bash

python3 ./setup.py develop --user

```

The matolotlib style can be installed in the sytlelib folder:

```bash

./install_mlp_style.sh

```

Usage
=====

After installation you can use the style with matplotlib:

```python

import matplotlib.pyplot as plt

plt.style.use('belle2')

```

  
To use the library you can do 

```python

import b2plot

b2plot.hist([1,2])

```

![Example](examples/example_stacked.pdf)

