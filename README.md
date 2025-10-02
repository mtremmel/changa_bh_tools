Code to analyze black hole data from ChaNGa galaxy formation simulations. It requires ```.BlackHoles``` and ```.BHMergers``` files for your simulation, along with a ```.param``` file.

<h3>Installation</h3>

clone this directory, ```cd``` into it and then install from source ```pip install .```

You should then be able to import the package in your current python environment with ```import bhtools```

<h3> Loading in Data </h3>

To load in all black hole data at once from a simulation you use the ```BHCatalog``` class. There are multiple ways to ensure you select the simulation you want, as outlined below:

1. If you are in the simulation directory already, where you have files of the form ```simname.000XXX```, ```simname.BHMergers```, ```simname.param```, etc then you can do ```bhcat = bhtools.BHCatalog()```

2. If you are outside the simulation directory and the directory ```path/to/simulation/data``` includes all the necessary files you can do ```bhcat = bhtools.BHCatalog('path/to/simulation/data')```

3. The two examples above will simply assume the first found file that ends with ```.BlackHoles``` is the one it wants (same for the other files it looks for). If you have multiple different simulations in the same directory (i.e. multiple different ```.BlackHoles``` files, then you can also specify the simname explicitly: ```bhcat = bhtools.BHCatalog('path/to/simulation/data',simname='simulation_number_1')```. This will fild files like ```path/to/simulation/data/simulation_number_1.BlackHoles``` to load in.

4. Within this catalog are two kinds of objects: ```bhcat.orbitdata``` that tracks every black hole's properties over time (as given in the ```.BlackHoles``` file) and ```bhcat.mergers``` that encodes information found in the ```.BHMergers``` file. You might have this information already stored in pickle files (especially if you are looking at Romulus data) so you can also create a catalog in this way without reading in the data anew:

```
import pickle
f = open('my_mergers_pickle.pkl','rb')
bhmerers_existing = pickle.load(f)
f.close()
f = open('my_orbit_pickle.pkl','rb')
bhorbit_existing = pickle.load(f)
f.close()
bhcat = bhtools.BHCatalog(bhorbit=bhorbit_existing, bhmergers=bhmergers_existing)
```

<h3>Using the Catalog</h3>

You extract information on black holes by treating the catalog like a dictionary which takes two arguments: the ID of the BH you care about and the data you want to extract.

```bhcat[<ID/iord of BH>, "data key"]``` will output the "data key" property for the BH of the given ID over time. You can see the available keys with ```bhcat.keys()```. This information all comes from the ```.BlackHoles``` file. If you were to select a BH which mergers and disappears from the simulation (every merger results in one of the two BHs being "eaten") then the output will start from the final time it exists until the time it forms.

You can add a third input into the above, either "raw" or "major". For example: ```bhcat[9074132513,'mass','major']``` will provide the mass of the BH ID 9074132513 tracking along its major progenitor branch (i.e. when it experiences a merger it decides which BH to follow based on the mass. If you were to do ```bhcat[9074132513,'iord','major']``` you will not necessary expect to see the same ID of the BH. Note, the ID number is stored under 'iord'. Similarly the "raw" keyword will follow the black hole's iord (which is also the default if no argument is given).



