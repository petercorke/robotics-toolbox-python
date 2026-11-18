"""The JupyterLite browser check is added to every kernel page of a built site."""

import importlib.util
from pathlib import Path

import pytest

LITE = Path(__file__).parent.parent / "docs" / "lite"


@pytest.fixture(scope="module")
def inject():
    spec = importlib.util.spec_from_file_location(
        "inject_jspi_check", LITE / "inject_jspi_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_site(root: Path, pages=("lab", "notebooks")) -> Path:
    for page in pages:
        (root / page).mkdir(parents=True)
        (root / page / "index.html").write_text(
            "<html>\n<head>\n<title>x</title></head>"
        )
    return root


def test_script_added_to_each_page(inject, tmp_path):
    site = make_site(tmp_path)
    changed = inject.inject(site)
    assert len(changed) == 2
    assert (site / "jspi-check.js").exists()
    for page in ("lab", "notebooks"):
        assert (
            '<script src="../jspi-check.js"></script>'
            in (site / page / "index.html").read_text()
        )


def test_running_twice_adds_it_once(inject, tmp_path):
    site = make_site(tmp_path)
    inject.inject(site)
    assert inject.inject(site) == []
    assert (site / "lab" / "index.html").read_text().count("jspi-check.js") == 1


def test_empty_site_is_an_error(inject, tmp_path):
    with pytest.raises(FileNotFoundError):
        inject.inject(tmp_path)
