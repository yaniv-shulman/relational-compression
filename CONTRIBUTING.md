# Contributing

Contributions to both the experiment code and the accompanying manuscript are welcome. Please open an issue before substantial work so that the proposed change can be discussed and scoped.

For questions or suggestions, please contact the corresponding author listed in the paper.

## Code Contributions

Keep changes focused on the maintained reproduction interface described in the [README](README.md). New experiments, configurations, metrics, or fixes should include appropriate tests and preserve reproducibility.

Before opening a pull request, run:

```bash
make check
```

Use the repository's existing conventions for type annotations, Google-style docstrings, named arguments, and explicit conditions. Avoid committing generated datasets, checkpoints, experiment outputs, or other artifacts under `data/` or `out/`.

## Paper Contributions

The manuscript master is `paper/relational_compression.tex`; section sources are in `paper/sections/` and figures are in `paper/graphics/`. Please keep manuscript edits narrowly scoped, preserve labels and citations unless a change requires them, and build the paper before submitting a pull request:

```bash
latexmk -cd -pdf paper/relational_compression.tex
```

For corrections to experiments or reported results, include the corresponding code, test, and manuscript changes together so that the paper remains reproducible from this repository.

## Pull Requests

Describe the motivation, affected studies or manuscript sections, and validation performed. Keep unrelated formatting or refactoring out of the same pull request. By contributing, you agree that your work may be distributed under the repository's applicable license terms.
