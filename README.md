# About Multi-physics-Network-Model (MpNM)
MpNM is a network model framework for simulating multi-physics processes (e.g. flow and heat) in porous media written in Python, which is developed by Zhejiang University and Imperial College London. This model can utilize is compatible with the pore network extraction algorithm developed by Imperial College London (https://github.com/ImperialCollegeLondon/pnextract). One of the major improvements in MpNM is to couple two networks (e.g. pore network and solid network dual-network model) to simulated couple mass and heat transfer in both pore space and solid phase. The source code is being prepared into different modules and will be uploaded continuously with the example datasets for demonstration.


## Example 1 Absolute permeability (Being upload)

This example is about predicting absolute permeability. There are four input network files (*_link1.dat, *_link2.dat, *_node1.dat, *_node2.dat come from pnextract) in folder ```sample_data/Bead_packing```. You can replace them to your files. 

* Locate the folder ```single_phase_flow```
```
python single_phase_permeability.py 
```
## Example 2 (Being upload)
This example is about heat and mass transfer in dual-network model.

