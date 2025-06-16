# Inhalt
Dieses Repository enthält ein Retrieval-Augmented Generation (RAG) System.

In diesem Projekt werden moderne NLP-Technologien mit einer Web- und API-basierten Schnittstelle kombiniert, um eine intelligente Suche und Beantwortung von Fragen zu Daten aus PDFs zu ermöglichen. Es besteht aus folgenden Hauptkomponenten:

- **Flask-basierter Webserver** mit Benutzeroberfläche und API-Endpunkten
- **MongoDB-Anbindung** zur Speicherung von Wissensartikeln und Vektoreinbettungen
- **Textvorverarbeitung & Embedding** via SpaCy & SentenceTransformers
- **Vektorbasierte Ähnlichkeitssuche** für semantisch relevante Dokumente
- **Einbettung & RAG-Logik** zur dynamischen Kontextzusammenstellung
- **LLM-Integration** (lokales Modell via GGUF) zur Antwortgenerierung
- **Automatisierte Tests mit LangSmith** für Testabdeckung und Validierung

Das System verarbeitet Textdaten, generiert Vektoreinbettungen, durchsucht relevante Dokumente mit Cosinus-Ähnlichkeit und verwendet ein Sprachmodell zur finalen Antworterzeugung. Zusätzlich gibt es eine UI für interaktive Abfragen.
