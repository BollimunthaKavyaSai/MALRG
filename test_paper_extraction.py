from paper_extraction import extract_papers_content


test_papers = [
    {
        "title": "Multi-Agent Systems and Blood Cell Formation",
        "authors": [
            "Nikolai Bessonov",
            "Ivan Demin",
            "Polina Kurbatova",
            "Laurent Pujo",
            "Vitaly Volpert",
        ],
        "year": 2013,
        "arxiv_id": "1309.1742v1",
        "url": "https://arxiv.org/abs/1309.1742v1",
        "doi": "10.5772/1936",
        "abstract": (
            "This is a test abstract used as a fallback "
            "if full-text extraction is unavailable."
        ),
        "source": "arXiv",
    }
]


print("\n" + "=" * 100)
print("PAPER EXTRACTION TEST")
print("=" * 100)

results = extract_papers_content(test_papers)

print("\n" + "=" * 100)
print("TEST RESULT")
print("=" * 100)

for paper in results:

    print(f"\nTitle          : {paper['title']}")
    print(f"Source         : {paper['content_source']}")
    print(f"PDF Path       : {paper['pdf_path']}")
    print(f"Text Length    : {paper['text_length']:,} characters")

    text = paper.get("full_text", "")

    print("\nFirst 1500 characters:")
    print("-" * 80)
    print(text[:1500])
    print("-" * 80)