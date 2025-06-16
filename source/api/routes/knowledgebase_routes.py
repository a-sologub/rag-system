from typing import Any, Mapping

import pymongo
from bson import ObjectId
from bson.json_util import dumps
from flask import Blueprint, request, current_app as app
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database

from source.preprocess.mongodb_to_vector_converter_script import update_vector_collection
from source.settings_loader import SettingsLoader

bp = Blueprint('knowledgebase', __name__)

USERNAME = ""
PASSWORD = ""


def set_global_variables(username, password):
    global USERNAME, PASSWORD
    USERNAME = username
    PASSWORD = password


@bp.route('/knowledgebase_data', methods=['POST'])
def get_knowledgebase() -> tuple[dict[str, str], int] | str:
    auth_data = request.get_json()

    settings = SettingsLoader(flask_app=app)
    set_global_variables(auth_data['username'], auth_data['password'])

    if not auth_data or not USERNAME or not PASSWORD:
        app.logger.warning("Login attempt with no data")
        return {"message": "No authentication data provided"}, 401

    client = pymongo.MongoClient(
        host=settings.get("mongodbConnectionSettings", "client"),
        username=USERNAME,
        password=PASSWORD,
        authSource=settings.get("mongodbConnectionSettings", "database"),
    )
    db = client[settings.get("mongodbConnectionSettings", "database")]
    collection = db[settings.get("mongodbConnectionSettings", "collectionKnowledgebase")]

    results = []
    try:
        for result in collection.find({}):
            results.append(result)
    except Exception as e:
        app.logger.error(e)
        return {"message": "Invalid credentials"}, 401

    return dumps(results)


@bp.route('/knowledgebase_update', methods=['POST'])
async def update_knowledgebase() -> tuple[dict[str, str], int]:
    data = request.get_json()

    client, db, knowledgebase_collection, vector_collection = get_mongodb_connection()
    update_knowledgebase_documents(knowledgebase_collection, data)
    await update_vector_collection(client, vector_collection, data)
    return {"message": "data in knowledgebase collection and vector collection updated"}, 200


def update_knowledgebase_documents(collection, data) -> list[UpdateOne]:
    bulk_operations = []

    for changes in data:
        # UpdateOne filter muss "_id" rein und der rest raus. aber format von _id is komisch. nochmal überarbeiten
        changes["_id"] = ObjectId(changes["_id"]["$oid"])

        bulk_operations.append(
            UpdateOne(
                {"title": changes["title"], "document_name": changes["document_name"], "page": changes["page"]},
                {"$set": {"revised_text": changes["revised_text"]}},
                upsert=False
            )
        )

    return collection.bulk_write(bulk_operations)


def get_mongodb_connection() -> tuple[
    MongoClient[Mapping[str, Any] | Any],
    Database[Mapping[str, Any] | Any],
    Collection[Mapping[str, Any] | Any],
    Collection[Mapping[str, Any] | Any]]:
    settings = SettingsLoader(flask_app=app)
    auth_db = settings.get("mongodbConnectionSettings", "database")
    client = MongoClient(settings.get("mongodbConnectionSettings", "client"), serverSelectionTimeoutMS=5000, username=USERNAME,
                         password=PASSWORD, authSource=auth_db)
    client.server_info()
    db = client[auth_db]
    knowledgebase_collection = db[settings.get("mongodbConnectionSettings", "collectionKnowledgebase")]
    vector_collection = db[settings.get("mongodbConnectionSettings", "collectionVector")]
    app.logger.debug("Connected to MongoDB successfully.")

    return client, db, knowledgebase_collection, vector_collection
