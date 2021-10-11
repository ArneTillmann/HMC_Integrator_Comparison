# Polymer model
To use visualize the sampled polymer conformations, use the functions in the `write_pdb.py` module. It requires the `csb` Python package(`pip install csb`).

For example:
```python
from write_pdb import write_ensemble, write_VMD_script

# X is your sample array
# X.shape = (n_samples, n_monomers, 3)
# r is the monomer radius: in the prior, you have a target distance between neighboring beads.
# So r is half that distance.

write_ensemble(X, "polymer_samples.pdb")
write_VMD_script("polymer_samples.pdb", r, X.shape[1], "vmd_script.rc")
```

The first function produces a file in the Protein Data Bank (https://www.rcsb.org/) format.
The second function produces a script which can be read by the Visual Molecular Dynamics (VMD, https://www.ks.uiuc.edu/Research/vmd/) program, which you will have to download and install.
You might need a Linux machine for that.
Once VMD is installed, type `vmd -e vmd_script.rc` and VMD should open with a visualization of the polymer. Use the little control window with the slider at the bottom to cycle through the states.
