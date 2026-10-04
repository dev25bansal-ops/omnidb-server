"""LEGAL-01: the declared license must match the LICENSE file that exists.

The defect this pins: `pyproject.toml` declared `license = { text =
"Apache-2.0" }` and `MANIFEST.in` had `include LICENSE`, yet no LICENSE file
existed. The distribution advertised Apache-2.0 on PyPI while carrying no
license text at all, and nothing failed. A declared license is a claim, not
proof -- these tests make the claim checkable.

The assertions here are deliberately about the *relationship* between the
metadata and the file, not about re-verifying Apache-2.0's wording. A license
file that is present but wrong, or metadata that points at a license the file
does not grant, is the failure mode that matters.
"""

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
LICENSE_FILE = REPO_ROOT / "LICENSE"
NOTICE_FILE = REPO_ROOT / "NOTICE"

# The first line of the canonical Apache-2.0 text, and the marker that its
# terms section both begin and end with. Checking the first line alone would
# accept a truncated file that stops after the header.
APACHE_FIRST_LINE = "Apache License"
APACHE_VERSION_LINE = "Version 2.0, January 2004"
APACHE_TERMS_HEADER = "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION"
APACHE_END_MARKER = "END OF TERMS AND CONDITIONS"
APACHE_APPENDIX = "APPENDIX: How to apply the Apache License to your work."


def _project_table() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]


def test_license_file_exists():
    """The whole defect in one assertion: this file simply has to be here."""
    assert LICENSE_FILE.is_file(), (
        f"{LICENSE_FILE} does not exist. pyproject.toml declares a license, so "
        "the distribution advertises terms it does not carry."
    )


def test_declared_license_is_pep639_spdx_string():
    """PEP 639 form only. The old table form validated nothing at all.

    `license = { text = "..." }` accepts any string, so it cannot detect a
    missing or mismatched file. Rejecting the table form here stops the
    deprecated, unverifiable spelling from creeping back in.
    """
    license_value = _project_table()["license"]
    assert isinstance(license_value, str), (
        "license must be a PEP 639 SPDX string, not a table. "
        f"Got {license_value!r} (type {type(license_value).__name__})."
    )
    assert license_value == "Apache-2.0"


def test_declared_license_files_are_present():
    """Every path in `license-files` must exist, or the build should have failed."""
    declared = _project_table()["license-files"]
    assert isinstance(declared, list), f"license-files must be a list, got {declared!r}"
    missing = [name for name in declared if not (REPO_ROOT / name).is_file()]
    assert not missing, f"license-files declares missing files: {missing}"


def test_license_files_includes_both_license_and_notice():
    """Apache-2.0 Section 4(d) imposes NOTICE propagation, so both ship."""
    declared = _project_table()["license-files"]
    assert "LICENSE" in declared
    assert "NOTICE" in declared


def test_license_file_is_the_full_apache_2_text():
    """Both sections, all nine clauses, and the appendix -- no truncation.

    A short file that merely starts with "Apache License" would pass a naive
    header check, so the end marker and the appendix are required too.
    """
    text = LICENSE_FILE.read_text(encoding="utf-8")
    assert APACHE_FIRST_LINE in text.splitlines()[0]
    assert APACHE_VERSION_LINE in text
    assert APACHE_TERMS_HEADER in text
    assert APACHE_END_MARKER in text
    assert APACHE_APPENDIX in text
    assert "limitations under the License." in text

    # All nine numbered clauses, 1 through 9, with none missing.
    clauses = {int(n) for n in re.findall(r"^   (\d+)\. ", text, re.MULTILINE)}
    assert clauses == set(range(1, 10)), f"expected clauses 1-9, found {sorted(clauses)}"

    # Section 4's four redistribution conditions (a)-(d). Clause (d) is the
    # NOTICE-propagation obligation, so it is the one most worth pinning.
    for letter in "abcd":
        assert f"({letter})" in text, f"redistribution condition ({letter}) missing"


def test_license_file_is_not_a_placeholder():
    """Guard against a stub that satisfies the header checks above."""
    text = LICENSE_FILE.read_text(encoding="utf-8")
    assert len(text) > 10_000, (
        f"LICENSE is only {len(text)} characters. The full Apache-2.0 text is "
        "~11.3k; a much shorter file is a placeholder, not a license."
    )
    # The canonical text is ASCII-only. Non-ASCII here would mean the file was
    # edited rather than copied, which is exactly the risk being guarded.
    text.encode("ascii"), "LICENSE should be plain ASCII"


def test_manifest_in_includes_license_and_notice():
    """MANIFEST.in is what puts these files in the sdist.

    `license-files` alone is not sufficient: it governs the wheel's metadata,
    while MANIFEST.in governs sdist contents.
    """
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for name in ("LICENSE", "NOTICE"):
        assert re.search(rf"^include {re.escape(name)}\b", manifest, re.MULTILINE), (
            f"MANIFEST.in has no `include {name}` line, so the sdist will not carry it."
        )


def test_notice_file_exists_and_states_apache_2():
    assert NOTICE_FILE.is_file(), f"{NOTICE_FILE} does not exist"
    notice = NOTICE_FILE.read_text(encoding="utf-8")
    assert "Apache License" in notice
    assert "Version 2.0" in notice


def test_no_deprecated_license_table_form_anywhere():
    """Belt-and-braces: the table form must not reappear in the metadata."""
    raw = PYPROJECT.read_text(encoding="utf-8")
    assert not re.search(r"license\s*=\s*\{\s*text\s*=", raw), (
        "found the deprecated `license = { text = ... }` table form"
    )


@pytest.mark.parametrize("filename", ["LICENSE", "NOTICE"])
def test_license_files_are_tracked_as_text_not_binary(filename):
    """A CRLF-mangled license still reads fine but breaks byte comparisons."""
    path = REPO_ROOT / filename
    raw = path.read_bytes()
    assert b"\r\n" not in raw, f"{filename} has CRLF line endings; normalize to LF"
    assert raw.endswith(b"\n"), f"{filename} should end with a newline"
    assert not raw.startswith(b"\xef\xbb\xbf"), f"{filename} has a UTF-8 BOM"
