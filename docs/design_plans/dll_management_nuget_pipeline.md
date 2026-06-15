# DLL Management: NuGet at build time + vendored native zip

Status: **implemented**. This documents the system as built.

## 1. Objective

Make it unambiguous **which C# DLLs a checkout uses**, and make **updating them a small,
reviewable, reproducible change** instead of swapping opaque 161 MB zips by hand.

The frequently-changing `VM.*` managed assemblies are acquired from the self-hosted NuGet feed at
the exact versions pinned in a committed lock file. The rarely-changing native Intel libraries and
the .NET framework runtime assemblies (which NuGet does not provide as loose DLLs) are kept in a
single committed zip. Maintainers assemble both into `src/videometer/DLLs/VM` and publish that as a
versioned, hash-verified bundle on a GitHub release.

**Distribution model (decided):** the wheel published to PyPI stays small (~0.04 MB) and ships only
`dlls.lock.json`. On first use of the `clr` backend, the package downloads the bundle recorded in
the lock and verifies its SHA256. This keeps the historical small-wheel PyPI workflow (the package
has always downloaded DLLs at first import) but upgrades it from a hardcoded `main`-branch URL to a
version-pinned, integrity-checked download. (Bundling all ~465 MB into the wheel was rejected: the
resulting ~162 MB wheel exceeds PyPI's 100 MB per-file limit.) End users need internet on first use
only; they never need the internal NuGet feed.

## 2. Problems this replaced

- Three opaque side-by-side ~161 MB zips (`DLLs.zip` committed; `DLLs_new.zip`/`DLLs_old.zip`
  local) with no manifest of contents or versions.
- Tests loaded an untracked, gitignored `DLLs/VM` folder that traced back to no known zip;
  "which DLLs are tests using?" was unanswerable.
- `setup_helper.setupDlls()` was dead code, hardcoded to download from the `main` branch.
- 161 MB binaries committed to git history.
- Six test modules each re-implemented (and had drifted in) the DLL path / pythonnet setup.

## 3. The DLL split (empirically determined)

The current `DLLs/VM` holds 74 DLLs. They divide cleanly:

| Source | Count | What |
|---|---|---|
| **NuGet** (`dlls.lock.json`) | 15 | 14 `VM.*` assemblies + `SixLabors.ImageSharp` |
| **Vendored zip** (`DLLs_vendored.zip`) | 59 | 55 native Intel libs (`ipp*`/`mkl*`/`libiomp5md`) + 4 .NET 8 desktop-runtime framework assemblies (`PresentationCore`, `WindowsBase`, `System.IO.Packaging`, `System.Drawing.Common`) |

Boundary rule: **if NuGet provides it as a loose redistributable DLL ⇒ fetch it; otherwise (native
Intel redistributable or in-box framework assembly) ⇒ vendor it in the zip.**

Notes from validation:
- Target framework is **net8.0-windows**; the feed is a self-hosted NuGet **v2 (OData)** feed
  (internal URL kept out of this public repo - see "Feed URL handling" below).
- `VM.*` inter-package dependency constraints are *minimums*, so all 14 are pinned explicitly;
  transitive resolution alone would pull older patch versions than those shipped.
- A full `dotnet publish` closure is much larger (~69 DLLs: `VM.Cloud`, WPF browser, Azure,
  SQLite, …). Only the 15-DLL allowlist is copied, keeping the output behavior-neutral.
- The assembled folder is **byte-identical** (74/74) to the previously committed `DLLs.zip`.
- `System.Drawing.Common` in the zip is the 8.0.15 desktop-runtime copy (1.5 MB), not the NuGet
  package (8.0.6, 0.6 MB) — hence it is vendored, not fetched.

## 4. Components (as built)

- **`nuget.config.template`** (committed) + **`nuget.config`** (gitignored, local) — declares the
  Videometer v2 feed and nuget.org. The real internal feed URL is kept out of this public repo;
  maintainers copy the template to `nuget.config` and fill in the URL. See "Feed URL handling".
- **`src/videometer/dlls.lock.json`** — source of truth: framework, feed, `vendored_zip`
  (file + sha256), `runtime_bundle` (file + url + sha256 of the downloadable bundle), the 15 pinned
  NuGet packages, and a `managed_assemblies` allowlist of the 15 DLLs (each with sha256).
- **`src/videometer/DLLs_vendored.zip`** (committed) — the 59 native + framework DLLs, stored
  under a `VM/` prefix so they extract straight into `DLLs/VM`.
- **`tools/fetch_dlls.py`** (maintainer/CI) — reads the lock; no-op if `DLLs/VM/.installed.json`
  matches; otherwise wipes `DLLs/VM`, extracts the vendored zip (verifying its sha256),
  `dotnet publish`es the pinned packages, copies + sha256-verifies the 15 allowlisted assemblies,
  writes the stamp. `--check` / `--force`; exposes `ensure_dlls()`. Needs the .NET SDK + the feed.
- **`tools/package_dll_bundle.py`** (maintainer) — zips the assembled `DLLs/VM` into a deterministic
  (reproducible-hash) `videometer-dlls-<version>.zip` and writes `runtime_bundle.{file,url,sha256}`
  back into the lock.
- **`tools/build_release.py`** (maintainer) — orchestrates: `fetch_dlls` → `--check` →
  `package_dll_bundle` → clean stale `build/`+egg-info → `python -m build` → verifies the wheel is
  small and contains no DLLs, then prints the release/upload commands.
- **`src/videometer/dll_provision.py`** (runtime, shipped) — `ensure_runtime_dlls()` downloads the
  bundle from `runtime_bundle.url` on first use, verifies its sha256, and extracts into `DLLs/VM`.
  No-op if a matching set is already present (detected by hashing the key assembly), so a developer
  who ran `fetch_dlls.py` is not forced to download. Needs only `requests` + a public URL.
- **`tests/conftest.py`** — single place that provisions DLLs (via `fetch_dlls.ensure_dlls()`), puts
  `DLLs/VM` on PATH / `add_dll_directory`, and loads pythonnet `coreclr` once. The 6 test modules
  (and `test_main`/`test_imageClass`) no longer carry DLL setup.
- **`python -m videometer --dll-info`** — prints the install stamp (framework, feed, vendored zip,
  pinned package versions). `--clean-dll` retained.
- **`src/videometer/vm_utils_clr.py`** — drops the no-op empty-`IPP2019Update1` PATH entry; calls
  `dll_provision.ensure_runtime_dlls()` (downloads on first use); puts `DLLs/VM` on PATH +
  `add_dll_directory`.
- **`pyproject.toml`** — `package-data` ships only `dlls.lock.json` (the wheel deliberately
  contains no DLLs).

## 5. Removed
- `src/videometer/DLLs.zip` (tracked, untracked via `git rm`; reconstructable byte-identically).
- `src/videometer/setup_helper.py` (dead GitHub-download path; nothing imported it).
- `.gitignore` now tracks `DLLs_vendored.zip`, ignores the assembled `DLLs/` and the scratch zips.
- `DLLs_new.zip` / `DLLs_old.zip` are local-only and left for the maintainer to delete manually.

> History note: removing the tracked `DLLs.zip` does not shrink existing `.git` history. An
> optional `git filter-repo` pass could reclaim it; out of scope here.

## 6. Releasing a new version (the workflow)
1. (If DLLs changed) edit versions in `src/videometer/dlls.lock.json`.
2. Run `python tools/build_release.py` (optionally `--version X.Y.Z`). It assembles + verifies the
   DLLs, builds the `videometer-dlls-<version>.zip` bundle, records its hash/url in the lock, and
   builds the small wheel.
3. Commit the updated `dlls.lock.json`.
4. Create the GitHub release `v<version>` and upload `dist/videometer-dlls-<version>.zip` as an
   asset at the URL recorded in the lock (the script prints the `gh release create` command).
5. `twine upload dist/videometer-<version>.whl`.

The bundle's hash is reproducible (deterministic zip), but **the bundle uploaded to the release must
be the one produced alongside the committed lock** — `build_release.py` produces both together.

## 7. Verification performed
- `fetch_dlls.py` assembles `DLLs/VM` byte-identical (74/74) to the old `DLLs.zip`; `--check` and
  the no-op rerun behave correctly.
- `package_dll_bundle.py` produces a reproducible bundle (same sha256 across rebuilds).
- The runtime download path was exercised end-to-end against a local HTTP server: a wiped install
  downloaded the bundle, passed the sha256 check, and extracted all 74 DLLs.
- The built wheel is ~0.04 MB and contains `dlls.lock.json` and no DLLs.
- Full test suite (260 passed, 37 skipped) runs through the single `conftest.py`, including the
  C#-read round-trip that confirms parity.

## 8. Feed URL handling (public repo)

This repository is public, so the internal NuGet feed URL is **not committed**:
- `nuget.config` is gitignored; `nuget.config.template` (with a placeholder) is committed.
  `tools/fetch_dlls.py` errors with copy-the-template instructions if `nuget.config` is missing.
- The `feed` field was removed from `dlls.lock.json`, the install stamp, and `--dll-info`, so the
  URL is not shipped in the wheel or the downloaded bundle either.
- End users never need the feed (they download the bundle from the public GitHub release).

Note this is information-hygiene, not access control: the feed itself should be internal-only
and/or authenticated. Removing the URL from the repo does not protect a feed that is reachable
and unauthenticated on the public internet.

## 9. Build/runtime requirements
- **Maintainers/CI** building a release need the .NET SDK and feed access (for `fetch_dlls.py`).
- **End users** need internet on first use of the `clr` backend only (to download the bundle); no
  .NET SDK and no feed access. The pure-`python` backend needs neither.
