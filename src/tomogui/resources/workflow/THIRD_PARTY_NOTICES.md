# Reused code and data

TomoGUI is distributed under GPL version 3; see LICENSE. The repository contains
source code and configurations derived from the TomoATT examples and the local
TomoATT research workflow. The TomoATT solver is installed separately and is
not bundled.

- TomoATT: https://github.com/TomoATT/TomoATT (GPL-3.0).
- Reused workspace code: preparation/conversion/scanning scripts, headless HDF5
  reader and expression engine, custom model builder, resolution scoring helpers,
  and data-free model/run templates. Source paths, original SHA-256 digests, and
  release adaptations are recorded in
  `src/tomogui/resources/workflow/provenance.json`.
- Packaged Slurm templates omit the original cluster account/partition and
  conda-environment activation. Scientific configuration defaults are preserved.
  The packaged JpGU settings module contains only the coefficient tuple read by
  TomoGUI. The original research catalogs, observation files, model volumes,
  inversion outputs and site-specific credentials are excluded.
- Natural Earth: the Taiwan outline is public-domain geographic data; see
  `src/tomogui/assets/README.txt` for its source. No event or station observations
  are embedded in that outline.
- PyTomoATT and the other Python dependencies are installed separately by pip
  and retain their own licenses. The bundled viewer uses its headless modules;
  Qt, VTK and PyVista are not required.
