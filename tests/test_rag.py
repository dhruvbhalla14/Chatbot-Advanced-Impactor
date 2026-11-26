import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from app.rag import (
    _contextual_query,
    _markdown_sections,
    bm25_search,
    clean_documents,
    source_payload,
    split_docs,
)


class RagPipelineTests(unittest.TestCase):
    def test_markdown_sections_preserve_heading(self):
        sections = _markdown_sections("# One\nText\n## Video Pipeline\nSegment clips.", "Overview")
        self.assertEqual(sections[1][0], "Video Pipeline")
        self.assertIn("## Video Pipeline", sections[1][1])

    def test_split_chunks_keep_source_metadata(self):
        source = Document(page_content="# Title\n\n" + ("Details about ball tracking. " * 100), metadata={"source": "company_knowledge/test.md", "section": "Tracking"})
        chunks = split_docs([source])
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.metadata["source"] == "company_knowledge/test.md" for chunk in chunks))
        self.assertTrue(all(chunk.metadata["section"] == "Tracking" for chunk in chunks))
        self.assertTrue(all(chunk.metadata["chunk_id"] for chunk in chunks))

    def test_cleaning_deduplicates_only_same_source_and_section(self):
        docs = [
            Document(page_content="Repeated technical detail.", metadata={"source": "a.md", "section": "Intro"}),
            Document(page_content="Repeated technical detail.", metadata={"source": "a.md", "section": "Intro"}),
            Document(page_content="Repeated technical detail.", metadata={"source": "b.md", "section": "Intro"}),
        ]
        self.assertEqual(len(clean_documents(docs)), 2)

    def test_bm25_finds_specific_terms(self):
        docs = [
            Document(page_content="MongoDB stores recovery information for queue jobs.", metadata={"source": "backend.md", "document_title": "Backend", "keywords": "MongoDB recovery queue"}),
            Document(page_content="Batting analysis reviews swing path and footwork.", metadata={"source": "batting.md", "document_title": "Batting", "keywords": "batting footwork"}),
        ]
        result = bm25_search("Where is recovery information stored for queue jobs?", docs, k=1)
        self.assertEqual(result[0].metadata["source"], "backend.md")

    def test_contextual_query_uses_prior_turn_for_followup(self):
        history = [{"role": "user", "content": "Explain CoachPlex."}, {"role": "assistant", "content": "It manages sessions and feedback."}]
        rewritten = _contextual_query("What about its schedule?", history)
        self.assertIn("CoachPlex", rewritten)
        self.assertIn("schedule", rewritten)

    def test_source_payload_exposes_status_and_section(self):
        doc = Document(page_content="x", metadata={"source": "doc.md", "document_title": "Doc title", "section": "Overview", "verification_status": "INTERNAL_CONTEXT", "source_urls": "[\"https://example.com\"]"})
        payload = source_payload([doc])
        self.assertEqual(payload[0]["title"], "Doc title")
        self.assertEqual(payload[0]["verification_status"], "INTERNAL_CONTEXT")
        self.assertEqual(payload[0]["urls"][0], "https://example.com")

    def test_system_prompt_requires_grounding_and_citations(self):
        from app.rag import SYSTEM_PROMPT
        self.assertIn("Do not invent facts", SYSTEM_PROMPT)
        self.assertIn("[S1]", SYSTEM_PROMPT)

    def test_malformed_cookie_docx_is_recovered_as_text(self):
        from pathlib import Path
        from app.rag import BASE_DIR, load_source

        docs = load_source(BASE_DIR / "data" / "04_Cookie_Policy.docx")
        self.assertTrue(docs)
        self.assertTrue(any(doc.page_content.strip() for doc in docs))
        self.assertEqual(docs[0].metadata["source"], "data/04_Cookie_Policy.docx")

    def test_every_knowledge_markdown_has_required_frontmatter(self):
        import yaml
        from app.rag import KNOWLEDGE_DIR, _read_frontmatter

        required = {"document_title", "category", "product", "source_type", "verification_status", "last_updated", "keywords"}
        statuses = {"VERIFIED_PUBLIC", "INTERNAL_CONTEXT", "HISTORICAL", "ANALYTICAL", "UNVERIFIED"}
        paths = list(KNOWLEDGE_DIR.rglob("*.md"))
        self.assertGreaterEqual(len(paths), 25)
        for path in paths:
            metadata, _ = _read_frontmatter(path.read_text(encoding="utf-8"))
            self.assertTrue(required.issubset(metadata), path.name)
            self.assertIn(metadata["verification_status"], statuses, path.name)

    def test_product_filter_limits_lexical_results(self):
        from app.rag import retrieve_documents

        class Store:
            index_to_docstore_id = {"one": "one", "two": "two"}

            class Docstore:
                values = {
                    "one": Document(page_content="CoachPlex manages coach schedules and sessions.", metadata={"source": "coach.md", "product": "coach_plex", "document_title": "CoachPlex", "keywords": "coach schedule sessions"}),
                    "two": Document(page_content="Advanced Impactor tracks cricket ball trajectory.", metadata={"source": "impact.md", "product": "advanced_impactor", "document_title": "Advanced Impactor", "keywords": "cricket ball trajectory"}),
                }

                def search(self, key):
                    return self.values[key]

            docstore = Docstore()

        result = retrieve_documents("coach schedules and sessions", Store(), filters={"product": "coach_plex"}, dense=False)
        self.assertTrue(result)
        self.assertTrue(all(doc.metadata["product"] == "coach_plex" for doc in result))

    def test_rate_limit_returns_cited_retrieved_passages(self):
        from app.rag import answer_question

        doc = Document(page_content="CoachPlex organizes coaches, athletes, sessions and feedback.", metadata={
            "source": "company_knowledge/products/coach_plex.md",
            "document_title": "CoachPlex",
            "section": "Public purpose",
            "product": "coach_plex",
            "verification_status": "VERIFIED_PUBLIC",
            "keywords": "coach sessions feedback",
            "source_urls": "[]",
        })

        class Store:
            index_to_docstore_id = {"one": "one"}

            class Docstore:
                def search(self, key):
                    return doc

            docstore = Docstore()

            def similarity_search(self, query, **kwargs):
                return [doc]

        class LimitedModel:
            def __init__(self, **kwargs):
                pass

            def invoke(self, prompt):
                raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")

        with patch("app.rag.ChatGoogleGenerativeAI", LimitedModel):
            response = answer_question("What does CoachPlex do?", Store())
        self.assertTrue(response["degraded"])
        self.assertIn("[S1]", response["answer"])
        self.assertEqual(response["sources"][0]["title"], "CoachPlex")


if __name__ == "__main__":
    unittest.main()

