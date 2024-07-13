"""
Analytical derivation of the stochastic output of a coincidence detector (CD) cell whose stochastic inputs behave as a
non-homogeneous Poisson process (NHPP) with both excitatory and inhibitory inputs.
================================================
Documentation is available in the docstrings and
online at https://github.com/nuniz/CoincidenceDetectionNetwork/blob/main/README.md.

Contents
--------
- `ei`: Function for excitatory-inhibitory interaction in neurons.
- `ee`: Function for excitatory-excitatory interaction in neurons.
- `simple_ee`: Simplified model of excitatory-excitatory interaction.
- `cd`: Function for modeling the output of a coincidence detector cell.
- `__version__`: Version string of the cd_network package.

Public API in the main TorchGating namespace
--------------------------------------
::
  ei              --- Function for excitatory-inhibitory interaction.
  ee              --- Function for excitatory-excitatory interaction.
  simple_ee       --- Simplified model of excitatory-excitatory interaction.
  cd              --- Function for modeling coincidence detector output.
  __version__     --- cd_network version string.

References
--------------------------------------
Krips R, Furst M. Stochastic properties of auditory brainstem coincidence detectors in binaural perception.
J Acoust Soc Am. 2009 Mar;125(3):1567-83. doi: 10.1121/1.3068446. PMID: 19275315.

"""

from .cells import cd, ee, ei, simple_ee
from .version import __version__
