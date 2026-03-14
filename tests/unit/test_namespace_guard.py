"""Guard test: ensure tests never import via the 'src.codebase_rag' namespace.

Both ``codebase_rag`` and ``src.codebase_rag`` are importable because
``pythonpath = . src`` is set in pytest.ini. However, they create
**separate** module objects, which makes ``unittest.mock.patch`` targets
unreliable. All tests must use the ``codebase_rag`` namespace exclusively.
"""

import sys


def test_no_src_codebase_rag_namespace_loaded() -> None:
    """Fail if any module under ``src.codebase_rag`` is loaded in the process."""
    offending = [name for name in sys.modules if name == "src.codebase_rag" or name.startswith("src.codebase_rag.")]
    assert not offending, f"Tests must not import from 'src.codebase_rag'. Found loaded modules: {offending}"
