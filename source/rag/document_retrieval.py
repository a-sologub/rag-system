"""Dieses Modul definiert die Funktion `search_similar_texts_in_db`, die für die Suche nach Dokumenten verantwortlich ist."""

import numpy as np
from flask import current_app as app
from sklearn.metrics.pairwise import cosine_similarity


async def search_similar_texts_in_db(query: str, top_k: int, full_text_content: bool) -> list[dict]:
    """Sucht nach Dokumenten in der Datenbank, die der Eingabeabfrage ähnlich sind.

    Diese Funktion führt die folgenden Schritte aus:
        1. Vorverarbeitung der Eingabeabfrage
        2. Generierung einer Einbettung für die vorverarbeitete Abfrage
        3. Abrufen aller Dokumenteinbettungen aus der Datenbank
        4. Berechnung der Cosinus-Ähnlichkeit zwischen der Abfrageeinbettung und den Dokumenteinbettungen
        5. Abrufen der top-k ähnlichsten Dokumente
        6. Optional Abrufen des vollständigen Textinhalts für die ähnlichen Dokumente

    Args:
        query: Die Eingabeabfragezeichenfolge, nach der ähnliche Dokumente gesucht werden sollen.
        top_k: Die Anzahl der ähnlichsten Dokumente, die abgerufen werden sollen. Standardmäßig 1.
        full_text_content: Ob der vollständige Textinhalt für ähnliche Dokumente abgerufen werden soll. Standardmäßig False.

    Returns:
        Eine Liste von Wörterbüchern, die Informationen über ähnliche Dokumente enthalten.

    Jedes Wörterbuch enthält:
        - document_name: Name des Dokuments
        - title: Titel des Dokuments
        - text: Textinhalt des Dokuments
        - similarity: Cosinus-Ähnlichkeitswert mit der Abfrage
        - outline_level: Gliederungsebene des Dokuments
        - outline_sublevel: Gliederungsunterebene des Dokuments
        - full_text: Vollständiger Textinhalt (wenn full_text_content True ist)

    Raises:
        RuntimeError: Wenn Probleme bei Datenbankoperationen auftreten.
        ValueError: Wenn die Einbettungsgenerierung fehlschlägt.
        Exception: Für alle anderen unerwarteten Fehler während der Ausführung.

    Hinweis:
        - Diese Funktion geht davon aus, dass 'db_manager', 'text_preprocessor' und 'vector_creator'
          in der Konfiguration der Flask-App verfügbar sind.
        - Die Funktion verwendet die Cosinus-Ähnlichkeit zur Messung der Dokumentähnlichkeit.
        - Dokumenteinbettungen werden gepoolt, um eine einzelne Vektordarstellung pro Dokument sicherzustellen.
    """
    app.logger.debug(
        f"Searching for similar texts. Query length: {len(query)}, top_k: {top_k}, full_text_content: {full_text_content}"
    )

    db_manager = app.config["db_manager"]
    text_preprocessor = app.config["text_preprocessor"]
    vector_creator = app.config["vector_creator"]
    llm_model = app.config["model"]

    query_list = await text_preprocessor.preprocess(query)
    preprocessed_query = " ".join(query_list)
    app.logger.debug(f"Preprocessed query length: {len(preprocessed_query)}")
    query_embedding = vector_creator.get_embedding(preprocessed_query)
    query_embedding = query_embedding.reshape(1, -1)

    # Abrufen aller Dokumenteinbettungen aus der Vektorsammlung
    all_docs = list(db_manager.vector_collection.find({}, {"knowledge_id": 1, "embeddings": 1}))
    app.logger.debug(f"Fetched {len(all_docs)} document embeddings from vector collection")

    doc_embeddings = np.array([doc["embeddings"] for doc in all_docs], dtype="float32")
    app.logger.debug(f"query_embedding shape: {query_embedding.shape}")
    app.logger.debug(f"doc_embeddings shape: {doc_embeddings.shape}")

    similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
    similarities_with_ids = [(doc["knowledge_id"], similarity) for doc, similarity in zip(all_docs, similarities)]
    sorted_similarities_with_ids = sorted(similarities_with_ids, key=lambda x: x[1], reverse=True)

    # Abrufen vollständiger Dokumente aus der Wissenssammlung
    results = []
    for doc_id, similarity in sorted_similarities_with_ids[:top_k]:
        app.logger.info(f"Knowledge ID {doc_id} with similarity {similarity:.4f}")
        full_doc = db_manager.knowledge_collection.find_one({"_id": doc_id})

        if full_doc:
            result = {
                "document_name": full_doc["document_name"],
                "title": full_doc["title"],
                "revised_text": text_preprocessor.delete_sensitive_data(str(full_doc["revised_text"])),
                "similarity": similarity.item(),
                "outline_level": full_doc["outline_level"],
                "outline_sublevel": full_doc["outline_sublevel"],
            }

            if full_text_content:
                # Abrufen aller Dokumente mit der gleichen outline_level
                same_level_docs = db_manager.knowledge_collection.find(
                    {
                        "document_name": full_doc["document_name"],
                        "outline_level": full_doc["outline_level"],
                    }
                ).sort("outline_sublevel")

                # Zusammenstellung des vollständigen Textinhalts
                full_text = [text_preprocessor.delete_sensitive_data(str(doc["revised_text"])) for doc in same_level_docs]
                max_content_length = app.config["settings"].get("documentRetrievalSettings", "maxContextLength")

                if len(llm_model.tokenizer.encode("".join(full_text))) <= max_content_length:
                    result["full_text"] = full_text
                    app.logger.debug(f"Fetched full text content for document {full_doc['document_name']}")
                else:
                    result["full_text"] = result["revised_text"]
                    app.logger.debug(f"Fetched only chunk content for document {full_doc['document_name']}")

            results.append(result)

    app.logger.debug(f"Returning {len(results)} similar documents")
    return results
