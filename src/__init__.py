"""ASTRA INTEL - defence document intelligence system.

The package is organised as a small, explicit RAG pipeline:

    pdf_processor -> chunker -> embeddings -> vector_store -> retriever
                                                     |
                              prompts -> llm -> grounding -> citations

Every stage is a plain, testable unit. ``pipeline.py`` wires them together and
is the only module that knows about the whole flow.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
