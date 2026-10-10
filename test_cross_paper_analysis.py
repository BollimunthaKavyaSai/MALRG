import pytest

from paper_extraction import extract_papers_content
from paper_analysis import analyze_paper
from cross_paper_analysis import analyze_cross_papers


TEST_PAPERS = [
    {
        "title": "Multi-Agent Systems and Blood Cell Formation",
        "authors": [
            "Nikolai Bessonov",
            "Ivan Demin",
            "Polina Kurbatova",
            "Laurent Pujo",
            "Vitaly Volpert"
        ],
        "year": 2013,
        "arxiv_id": "1309.1742v1",
        "url": "https://arxiv.org/abs/1309.1742v1",
        "doi": "10.5772/1936",
        "abstract": (
            "Mathematical modelling of hematopoiesis "
            "using multi-agent systems."
        ),
        "source": "arXiv"
    },
    {
        "title": (
            "Verification & Validation of Agent Based "
            "Simulations using the VOMAS approach"
        ),
        "authors": [
            "Muaz A. Niazi",
            "Amir Hussain",
            "Mario Kolberg"
        ],
        "year": 2017,
        "arxiv_id": "1708.02361v1",
        "url": "https://arxiv.org/abs/1708.02361v1",
        "doi": None,
        "abstract": (
            "Verification and validation of agent-based "
            "simulations using a virtual overlay "
            "multi-agent system."
        ),
        "source": "arXiv"
    },
    {
        "title": (
            "Automatic Verification of Parameterised "
            "Interleaved Multi-Agent Systems"
        ),
        "authors": [
            "Panagiotis Kouvaros",
            "Alessio Lomuscio"
        ],
        "year": 2013,
        "arxiv_id": "1301.6431v2",
        "url": "https://arxiv.org/abs/1301.6431v2",
        "doi": None,
        "abstract": (
            "Automatic verification of parameterised "
            "interleaved multi-agent systems."
        ),
        "source": "arXiv"
    }
]


# Tests the Cross-Paper Analysis agent with several real test papers: extracts their full text, analyses each
# paper, then calls analyze_cross_papers() on them and checks the structure of the comparison result.
def test_cross_paper_analysis():

    print("\n" + "=" * 80)
    print("CROSS-PAPER ANALYSIS V2 TEST")
    print("=" * 80)

    print("\nExtracting papers...")

    extracted = extract_papers_content(TEST_PAPERS)

    assert extracted
    assert len(extracted) == len(TEST_PAPERS)

    print(f"Extracted papers: {len(extracted)}")

    analyzed_papers = []

    for index, paper in enumerate(extracted, start=1):

        print(
            f"\nAnalyzing paper {index}/{len(extracted)}: "
            f"{paper.get('title', 'Unknown')}"
        )

        analysis = analyze_paper(paper)

        assert isinstance(analysis, dict)

        paper_with_analysis = dict(paper)
        paper_with_analysis["paper_analysis"] = analysis

        analyzed_papers.append(paper_with_analysis)

    assert len(analyzed_papers) == 3

    print("\nRunning cross-paper synthesis...")

    result = analyze_cross_papers(analyzed_papers)

    assert isinstance(result, dict)

    required_fields = [
        "overall_synthesis",
        "common_themes",
        "theme_evidence",
        "methodology_comparison",
        "comparison_evidence",
        "finding_comparison",
        "agreements",
        "disagreements",
        "research_trends",
        "cross_paper_limitations",
        "research_gaps",
        "gap_evidence",
        "future_research_directions"
    ]

    for field in required_fields:
        assert field in result, f"Missing field: {field}"

    assert isinstance(result["overall_synthesis"], str)
    assert isinstance(result["common_themes"], list)
    assert isinstance(result["theme_evidence"], list)
    assert isinstance(result["methodology_comparison"], list)
    assert isinstance(result["comparison_evidence"], list)
    assert isinstance(result["research_gaps"], list)
    assert isinstance(result["gap_evidence"], list)
    # A functional cross-paper synthesis must contain
    # actual generated content, not only the expected schema.
    assert result["overall_synthesis"].strip(), (
        "Cross-paper synthesis is empty"
    )

    synthesis_fields = [
        result["common_themes"],
        result["theme_evidence"],
        result["methodology_comparison"],
        result["comparison_evidence"],
        result["finding_comparison"],
        result["research_gaps"],
        result["gap_evidence"]
    ]

    assert any(synthesis_fields), (
        "Cross-paper synthesis contains no thematic, "
        "comparative, or gap information"
    )

    print("\n" + "=" * 80)
    print("CROSS-PAPER V2 STRUCTURE")
    print("=" * 80)

    print(
        f"Overall synthesis: "
        f"{len(result['overall_synthesis'])} characters"
    )

    print(
        f"Common themes: "
        f"{len(result['common_themes'])}"
    )

    print(
        f"Theme evidence records: "
        f"{len(result['theme_evidence'])}"
    )

    print(
        f"Methodology comparisons: "
        f"{len(result['methodology_comparison'])}"
    )

    print(
        f"Comparison evidence records: "
        f"{len(result['comparison_evidence'])}"
    )

    print(
        f"Research gaps: "
        f"{len(result['research_gaps'])}"
    )

    print(
        f"Gap evidence records: "
        f"{len(result['gap_evidence'])}"
    )

    if result["theme_evidence"]:
        for item in result["theme_evidence"][:3]:
            assert isinstance(item, dict)
            assert "theme" in item
            assert "paper_ids" in item
            assert "supporting_evidence" in item

    if result["comparison_evidence"]:
        for item in result["comparison_evidence"][:3]:
            assert isinstance(item, dict)
            assert "comparison" in item
            assert "paper_ids" in item
            assert "evidence" in item

    if result["gap_evidence"]:
        for item in result["gap_evidence"][:3]:
            assert isinstance(item, dict)
            assert "gap" in item
            assert "paper_ids" in item
            assert "basis" in item

    print("\nCross-paper V2 validation passed.")

