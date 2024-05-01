# JAXOvercooked
Implementation of multiple Multi-Agent Reinforcement Learning (MARL) methods on the overcooked environment using JAX.

## IPPO Implementation
To run the IPPO implementation, first create a virtual environment 
``` python -m venv .venv```

Second, activate this virtual environment 
On Linux: 
``` source .venv/bin/activate```
On Windows:
``` .venv\Scripts\activate.bat```

Then, install the requirements 
``` pip install -r requirements.txt```

After this, the IPPO implementation can be run by:
```cd path/to/JAXOvercooked 
python -m baselines.ippo_ff_overcooked```


