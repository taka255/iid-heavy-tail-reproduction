# Reproduction code

This code compares the theoretical learning curve obtained by numerically solving the saddle-point equations with experiments.

## Usage

```sh
cd reproduction
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. --threads=auto figure1a.jl
```

Outputs are written to `output/`.
