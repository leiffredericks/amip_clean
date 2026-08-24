# AMIP clean processing

Path-based tools for preparing AMIP climate-model data while keeping large
NetCDF inputs and temporary products off the local Git repository.

## What belongs in Git

- Python processing and catalog code
- Small text manifests describing remote data layouts
- Configuration, documentation, and tests

## What does not belong in Git

`data/` is intentionally ignored. Use it only as a local or remote scratch
location for NetCDF inputs, intermediate processing files, and final data
products. This keeps repository transfers fast and prevents large files from
being added accidentally.

## Maria catalog

`build_maria_catalog.py` reads a text-only remote file manifest and builds a
catalog of complete `(model, experiment, member)` bundles. A bundle is ready
for spatial net-radiation processing when it contains `tas`, `rsdt`, `rlut`,
and `rsut`.

Regenerate the catalog after replacing or updating the manifest:

```bash
python3 build_maria_catalog.py remote_structure_manifest_maria.txt \
  --output-dir maria_catalog
```

The resulting `maria_processing_tasks.json` is the task list for the future
one-member-at-a-time processing pipeline. It contains file paths only and does
not transfer or open the NetCDF data.

## Moving between local and remote machines

Push this repository to a private Git remote, then clone or pull it on the
machine that has access to the climate archive. Keep data paths machine-specific
and outside version control; the scripts should receive those paths as command
line arguments or configuration values.

Before committing, check that no data was staged:

```bash
git status
```
