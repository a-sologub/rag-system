"""
Dieses Utility konvertiert Text aus MongoDB-Objekten in eine Vektorsammlung.
"""

from flask import current_app as app
from pymongo import UpdateOne
from pymongo.collection import Collection


async def create_vector_representation(
        vector_collection: Collection,
        data
):
    pipeline = [{"$match": {"knowledge_id": {"$in": [obj["_id"] for obj in data]}}}]

    results = list(vector_collection.aggregate(pipeline))
    bulk_operations = []

    for doc in results:
        text_preprocessor = app.config["text_preprocessor"]
        vector_creator = app.config["vector_creator"]

        knowledge_object = next(filter(lambda obj: obj["_id"] == doc.get("knowledge_id"), data), None)
        text = knowledge_object.get("revised_text")

        processed_text = await text_preprocessor.preprocess(text, False)
        embedding = vector_creator.get_embedding(processed_text).tolist()

        bulk_operations.append(
            UpdateOne(
                {"title": doc["title"], "document_name": doc["document_name"], "page": doc["page"]},
                {"$set": {"embeddings": embedding}},
                upsert=False
            )
        )

    try:
        bulk_result = vector_collection.bulk_write(bulk_operations)
        app.logger.debug(
            f"Updated embeddings."
        )
        return bulk_result
    except Exception as e:
        app.logger.error(
            f"Error updating embeddings: {e}"
        )


async def update_vector_collection(client, vector_collection, data):
    await create_vector_representation(
        vector_collection,
        data
    )

    client.close()
    app.logger.debug("\nProcessing completed.")
