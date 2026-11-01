# Git-style releases

`main` is the development branch. Release versions follow Semantic Versioning;
the Git tag is `vMAJOR.MINOR.PATCH`. A tag identifies an immutable source snapshot.
GitHub releases attach a wheel, source distribution and SHA-256 checksums. No
automatic PyPI upload is configured.

For each release:

1. Update `pyproject.toml`, `src/tomogui/__init__.py`, the matching CHANGELOG
   section, and versioned installation examples. Refresh bundled workflow code
   only when intended using `python release/bundle_workflow.py --source WORKSPACE`.
   Review provenance, upstream license obligations and the data-free allowlist.
2. Run the tests and build locally, inspect the distributions, and commit the
   changes. Push `main` and wait for CI to pass.
3. Run `python release/check_version.py --tag vX.Y.Z`.
4. Create and push an annotated tag:

   ```bash
   git tag -a vX.Y.Z -m "TomoGUI vX.Y.Z"
   git push origin vX.Y.Z
   ```

5. The Release workflow verifies the tag/version/changelog, reruns tests, builds
   wheel/source assets, computes SHA256SUMS and creates the GitHub release.
   Verify the workflow and download links before announcing it. Mark explicit
   prerelease tags/releases appropriately when introducing a prerelease scheme.

Do not move or overwrite a published tag. Fix released defects with a new patch
version. Existing initialized projects intentionally retain their helper snapshot;
creating a fresh project is the supported way to adopt a new template snapshot.

The initial v0.1.0 release is alpha. Solver execution on production MPI/HPC is
outside the automated release tests.
