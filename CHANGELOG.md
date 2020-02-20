# Changelog

This document only describes _breaking_ changes in meshio. If you are interested in bug
fixes, enhancements etc., best follow [the meshio project on
GitHub](https://github.com/nschloe/meshio).

## v4.0.0 (Feb 18, 2020)

- `mesh.cells` used to be a dictionary of the form
  ```python
  {
    "triangle": [[0, 1, 2], [0, 2, 3]],
    "quad": [[0, 7, 1, 10], ...]
  }
  ```
  From 4.0.0 on, `mesh.cells` is a list of tuples,
  ```python
  [
    ("triangle", [[0, 1, 2], [0, 2, 3]]),
    ("quad", [[0, 7, 1, 10], ...])
  ]
  ```
  This has the advantage that multiple blocks of the same cell type can be accounted
  for. Also, cell ordering can be preserved.

  You can now use the method `mesh.get_cells_type("triangle")` to get all cells of
  `"triangle"` type, or use `mesh.cells_dict` to build the old dictionary structure.

- `mesh.cell_data` used to be a dictionary of the form
  ```python
  {
    "triangle": {"a": [0.5, 1.3], "b": [2.17, 41.3]},
    "quad": {"a": [1.1, -0.3, ...], "b": [3.14, 1.61, ...]},
  }
  ```
  From 4.0.0 on, `mesh.cell_data` is a dictionary of lists,
  ```python
  {
    "a": [[0.5, 1.3], [1.1, -0.3, ...]],
    "b": [[2.17, 41.3], [3.14, 1.61, ...]],
  }
  ```
  Each data list, e.g., `mesh.cell_data["a"]`, can be `zip`ped with `mesh.cells`.

  An old-style `cell_data` dictionary can be retrieved via `mesh.cell_data_dict`.
