"""Legal-structure parsing. The first group uses synthetic pages (no source files needed) and covers every extractor layout;
the second group checks the real EU AI Act index and needs the source documents."""
import pytest

import rag

SRC = {"source_id": "AIA", "file": "synthetic.pdf", "version": "t", "authority": 1, "url": None}


def _pages(layout):
    """A miniature OJ-style regulation. layout: 'inline' (pypdf 6+), 'split' (PyMuPDF: markers on their own line) or 'kerned' (pypdf 5: 'Ar ticle', 'HA VE')."""
    art = "Ar ticle" if layout == "kerned" else "Article"
    formula = "HA VE ADOPTED THIS REGUL A TION:" if layout == "kerned" else "HAVE ADOPTED THIS REGULATION:"

    def para(n, text):
        return [f"{n}.", text] if layout == "split" else [f"{n}. {text}"]

    def point(x, text):
        return [f"({x}) {text}"]
    lines = ["(1) First recital text.", "(2) Second recital text.", formula, "CHAPTER I", "GENERAL PROVISIONS", f"{art} 1", "Subject matter"]
    lines += para(1, "The purpose of this Regulation is to improve the functioning of the internal market.")
    lines += para(2, "This Regulation lays down rules.")
    lines += [f"{art} 2", "Scope"] + para(1, "This Regulation applies to providers placing AI systems on the market.")
    lines += [f"{art} 3", "Definitions", "For the purposes of this Regulation:", "(1) ‘AI system’ means a machine-based system."]
    lines += [f"{art} 4", "AI literacy"] + para(1, "Providers shall take measures.") + [f"{art} 5", "Prohibited AI practices"] + para(1, "The following AI practices shall be prohibited:")
    lines += [f"{art} 6", "Classification rules for high-risk AI systems"]
    lines += para(1, "That AI system shall be considered to be high-risk where both of the following conditions are fulfilled:")
    lines += point("a", "the AI system is intended to be used as a safety component;") + point("b", "the product is required to undergo a conformity assessment.")
    lines += para(2, "AI systems referred to in Annex III shall be considered to be high-risk.")
    lines += para(3, "By derogation from paragraph 2, an AI system shall not be considered to be high-risk where it performs a narrow procedural task.")
    lines += ["ANNEX I", "List of Union harmonisation legislation", "1. Directive 2006/42/EC"]
    return ["\n".join(lines)]


@pytest.mark.parametrize("layout", ["inline", "split"])
def test_numbered_paragraphs_stay_numbered_for_every_extractor_layout(layout):
    pages = [rag._canon_page(p) for p in _pages(layout)]
    refs = [c["ref"] for c in rag.chunk_aia(SRC, pages)]
    assert "Art. 6(1)" in refs and "Art. 6(2)" in refs and "Art. 6(3)" in refs, refs
    assert not any(r.startswith("Art. 6(a)") or r.startswith("Art. 6(c)") for r in refs), refs  # numbered paragraphs are never read as lettered points
    assert "Art. 1(1)" in refs and "Art. 2(1)" in refs and "Recital (2)" in refs and any(r.startswith("Annex I") for r in refs)


def test_kerned_extractor_output_is_rejected_loudly():
    with pytest.raises(rag.SourceError, match="PyMuPDF"):
        rag.chunk_aia(SRC, [rag._canon_page(p) for p in _pages("kerned")])


def test_canon_page_joins_lone_markers():
    assert rag._canon_page("1.\nThe purpose\n(a)\nthe system\n(12)\ntext") == "1. The purpose\n(a) the system\n(12) text"


def test_missing_enacting_formula_is_a_loud_error_not_a_stop_iteration():
    with pytest.raises(rag.SourceError):
        rag.chunk_aia(SRC, ["Article 1\nSubject matter\n1. Text."])


@pytest.mark.needs_sources
def test_real_ai_act_index_has_the_expected_references(index):
    refs = {c["ref"] for c in index.chunks if c["source_id"] == "AIA"}
    for r in [f"Art. 5(1)({x})" for x in "abcdefgh"] + ["Art. 6(1)", "Art. 6(2)", "Art. 6(3)", "Art. 6(4)", "Art. 2(1)", "Art. 14(4)", "Art. 43(1)", "Art. 50(1)"]:
        assert r in refs, r
    assert not {"Art. 6(a)", "Art. 6(b)", "Art. 6(c)", "Art. 6(d)"} & refs
    assert sum(1 for c in index.chunks if c["source_id"] == "AIA") > 900
    assert len(index.chunks) > 1200  # footnotes and non-English pages are no longer indexed (release audit), so the count is lower than before
    index.validate()  # raises SourceError if an article, annex or mandatory anchor is missing


@pytest.mark.needs_sources
def test_validation_fails_loudly_when_a_mandatory_anchor_is_missing(index, monkeypatch):
    chunks = [c for c in index.chunks if not (c["source_id"] == "AIA" and c["ref"].startswith("Art. 6(3)"))]
    monkeypatch.setattr(index, "chunks", chunks)
    with pytest.raises(rag.SourceError, match="Art. 6\\(3\\)"):
        index.validate()
