from paper_extraction import extract_papers_content
from paper_analysis import analyze_paper

TEST_PAPER = {
    "title": "Multi-Agent Systems and Blood Cell Formation",
    "authors": ["Nikolai Bessonov", "Ivan Demin", "Polina Kurbatova", "Laurent Pujo", "Vitaly Volpert"],
    "year": 2013,
    "arxiv_id": "1309.1742v1",
    "url": "https://arxiv.org/abs/1309.1742v1",
    "doi": "10.5772/1936",
    "abstract": "This paper discusses mathematical modelling of hematopoiesis using multi-agent systems.",
    "source": "arXiv"
}


# Tests the Paper Analysis agent on one real test paper.
# Step 1 downloads the PDF and extracts its text; step 2 runs analyze_paper() and asserts that the result is a
# dictionary containing 'paper_analysis', then prints the extracted evidence fields.
def test_paper_analysis():
    print()
    print("=" * 100)
    print("SECTION 2.2 - PAPER ANALYSIS AGENT TEST")
    print("=" * 100)

    print()
    print("Step 1: Extracting full paper...")
    extracted = extract_papers_content([TEST_PAPER])
    assert extracted, "Paper extraction returned no papers."
    paper = extracted[0]
    print(f"Content source : {paper.get('content_source')}")
    print(f"Text length   : {paper.get('text_length', 0):,} characters")
    assert paper.get("text_length", 0) > 0

    print()
    print("Step 2: Running Paper Analysis Agent...")
    result = analyze_paper(paper)
    assert isinstance(result, dict)
    assert "paper_analysis" in result
    analysis = result["paper_analysis"]

    print()
    print("PAPER ANALYSIS RESULT")
    print("=" * 100)

    fields = [
        "research_problem",
        "objectives",
        "methodology",
        "dataset",
        "models_and_techniques",
        "evaluation_metrics",
        "key_findings",
        "contributions",
        "limitations",
        "future_work"
    ]

    for field in fields:
        print()
        print(f"{field}:")
        value = analysis.get(field, [])
        if isinstance(value, list):
            for item in value:
                print(f"  - {item}")
        else:
            print(f"  {value}")

    print()
    print("Evidence Records:")
    evidence_records = analysis.get("evidence_records", [])
    assert isinstance(evidence_records, list)
    for index, record in enumerate(evidence_records, start=1):
        assert isinstance(record, dict)
        print(f"  {index}. [{record.get('evidence_type', '')}] {record.get('claim', '')}")
        print(f"     Evidence : {record.get('evidence', '')}")
        print(f"     Section  : {record.get('section', '')}")
        print(f"     Page     : {record.get('page')}")
        print(f"     Chunk    : {record.get('chunk')}")
        print(f"     Status   : {record.get('status', '')}")

    print()
    print("Coverage:")
    coverage = analysis.get("coverage", {})
    assert isinstance(coverage, dict)
    for key, value in coverage.items():
        print(f"  {key}: {value}")

    required_fields = [
        "research_problem", "objectives", "methodology", "dataset",
        "models_and_techniques", "evaluation_metrics", "key_findings",
        "contributions", "limitations", "future_work",
        "evidence_records", "coverage"
    ]

    for field in required_fields:
        assert field in analysis, f"Missing V2 field: {field}"

    print()
    print("SECTION 2.2 TEST COMPLETE")
    print("=" * 100)
