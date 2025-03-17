# Homework 2

The folder structure should be

```text
.
├── in/
│   ├── 4/
│   └── 5/
├── out/
│   ├── 4/
│   └── 5/
├── src/
│   ├── algo.py
│   ├── func.py
│   └── util.py
├── main.py
└── README.md
```

The only required libraries are NumPy and Matplotlib.
To start, run `main.py`.
The input information for problem 4 is found in `/in/4/` and similarly for problem 5, in `/in/5/`.

- `algo.py` contains the main algorithms: proximal gradient, accelerated proximal gradient, and alternating directions method of multipliers.
- the ADMM variant for problem 5 is flagged by `var=True`.
- `func.py` contains proximal operators and stopping functions for problems 4 and 5.
- `util.py` contains functions for input creation, array saving, and visualization.
