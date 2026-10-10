from query_understanding import (
    understand_research_query
)


TEST_TOPICS = [
    "multi-agent systems healthcare",
    "multi-agent systems financial fraud detection",
    "multi-agent systems cybersecurity",
    "explainable AI autonomous vehicles"
]


print("\n")
print("=" * 100)
print("QUERY UNDERSTANDING AGENT TEST")
print("=" * 100)


for number, topic in enumerate(
    TEST_TOPICS,
    start=1
):

    print(
        f"\n\n{'=' * 100}"
    )

    print(
        f"TEST {number}: {topic}"
    )

    print(
        f"{'=' * 100}"
    )

    result = understand_research_query(
        topic
    )

    print("\nResearch Intent:")
    print(
        result["research_intent"]
    )

    print("\nCore Concepts:")
    for item in result["core_concepts"]:
        print(
            f"  • {item}"
        )

    print("\nDomain:")
    for item in result["domain"]:
        print(
            f"  • {item}"
        )

    print("\nSubtopics:")
    for item in result["subtopics"]:
        print(
            f"  • {item}"
        )

    print("\nRelated Concepts:")
    for item in result["related_concepts"]:
        print(
            f"  • {item}"
        )

    print("\nSearch Queries:")
    for index, query in enumerate(
        result["search_queries"],
        start=1
    ):
        print(
            f"  {index}. {query}"
        )


print(
    "\n\n" + "=" * 100
)

print(
    "QUERY UNDERSTANDING TEST COMPLETE"
)

print(
    "=" * 100
)