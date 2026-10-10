from app import ResearchState, search_agent_node

state = ResearchState(
    topic="multi-agent systems healthcare",
    max_papers=5
)

result = search_agent_node(state)

print("\n" + "=" * 70)
print("SEARCH TEST RESULT")
print("=" * 70)

papers = result.get("raw_papers_list", [])

print(f"Retrieved papers: {len(papers)}")

for i, paper in enumerate(papers, 1):
    print(f"\n{i}. {paper['title']}")
    print(f"   Authors : {', '.join(paper['authors'])}")
    print(f"   Year    : {paper['year']}")
    print(f"   ID      : {paper['paper_id']}")
    print(f"   DOI     : {paper['doi']}")
    print(f"   Source  : {paper['source']}")