"""
Dieses Utility konvertiert Text und Bilder aus PDF-Dateien in eine MongoDB-Datenbank.

Es extrahiert Inhalte aus PDFs, ohne Kopf- und Fußzeilen, organisiert sie nach der Gliederung des Dokuments und
verwendet die Gliederungsebenen aus der PDF-Struktur.
"""

import glob
import logging
import os
import sys
import argparse
from argparse import Namespace
from pathlib import Path
from typing import List, Dict, Tuple, Union

import pymupdf
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database
from tqdm import tqdm

VALID_AFFIRMATIVE_INPUTS = {"y", "yes", ""}
VALID_NEGATIVE_INPUTS = {"n", "no"}


def initialize_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s\t%(levelname)s\t(TID %(thread)d %(threadName)s)\t%(funcName)s:%(lineno)d\t%(message)s",
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def print_description() -> None:
    logging.info("""
    This utility is designed to convert text and images from PDFs files to MongoDB database.
    It extracts content, adds outline levels, and organizes the data.
    For this utility to work, you will need to maintain a folder with PDFs files and parameters for connection to MongoDB database.
    """)


def request_continuation() -> None:
    while True:
        program_continuation = input("Continue [Y/n]: ").strip().lower()
        if program_continuation in VALID_NEGATIVE_INPUTS:
            sys.exit()
        if program_continuation in VALID_AFFIRMATIVE_INPUTS:
            break
        else:
            logging.info("Sorry, I didn't understand your input. Please try again.")


def get_pdfs_file_path() -> List[str]:
    pdfs_files_path = input("Enter the folder path containing the PDFs files: ")
    pdfs_files = glob.glob(os.path.join(pdfs_files_path, "*.pdf"))
    if len(pdfs_files) > 0:
        logging.info("Found the following PDF files: \n")
        for pdf_file in pdfs_files:
            logging.info(Path(f"{pdf_file}\n").stem)
        return pdfs_files
    else:
        logging.info(f"No PDF files were found in '{pdfs_files_path}'.")


def extract_text_and_images_by_outline_excluding_headers_footers(
        pdf_path: str,
        header_height: int,
        footer_height: int
) -> Tuple[Dict[Tuple[str, int, int, int], Tuple[str, List[Dict]]], str]:
    document = pymupdf.open(pdf_path)
    document_name = document.metadata.get("title", Path(pdf_path).stem)
    outlines = document.get_toc()
    text_by_outline = {}

    if not outlines:
        text, images = extract_text_and_images_for_section(
            document, 0, len(document), header_height, footer_height
        )
        first_line = text.strip().split("\n", 1)[0] if text.strip() else "Untitled"
        text_by_outline[(first_line, 1, 0, 0)] = (text.strip(), images)
    else:
        for i, outline in enumerate(outlines):
            level, title, page_num = outline
            page_num -= 1

            next_outline_page_num = None
            for next_outline in outlines[i + 1:]:
                if next_outline[0] <= level:
                    next_outline_page_num = next_outline[2] - 1
                    break

            if next_outline_page_num is None:
                end_page = len(document)
            else:
                end_page = next_outline_page_num

            text, images = extract_text_and_images_for_section(
                document, page_num, end_page, header_height, footer_height
            )
            outline_level = level
            outline_sublevel = i + 1  # Use the index as a sublevel
            text_by_outline[(title, page_num + 1, outline_level, outline_sublevel)] = (text.strip(), images)

    document.close()
    return text_by_outline, document_name


def extract_text_and_images_for_section(
        document: pymupdf.Document,
        start_page: int,
        end_page: int,
        header_height: int,
        footer_height: int,
        min_image_size: int,
) -> Tuple[str, List[Dict]]:
    text = ""
    images = []
    image_index = 1

    for page_num in range(start_page, end_page):
        page = document.load_page(page_num)
        page_rect = page.rect
        header_rect = pymupdf.Rect(
            page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y0 + header_height
        )
        footer_rect = pymupdf.Rect(
            page_rect.x0, page_rect.y1 - footer_height, page_rect.x1, page_rect.y1
        )

        for block in page.get_text("dict")["blocks"]:
            block_rect = pymupdf.Rect(block["bbox"])
            if block["type"] == 0 and not (
                    header_rect.intersects(block_rect) or footer_rect.intersects(block_rect)
            ):
                for line in block["lines"]:
                    for span in line["spans"]:
                        text += span["text"]
                text += "\n"

        for img in page.get_images(full=True):
            xref = img[0]
            base_image = document.extract_image(xref)
            if base_image:
                img_rect = page.get_image_bbox(img)
                if not (header_rect.intersects(img_rect) or footer_rect.intersects(img_rect)):
                    if img_rect.width >= min_image_size or img_rect.height >= min_image_size:
                        images.append(
                            {
                                "image_bytes": base_image["image"],
                                "image_ext": base_image["ext"],
                                "image_order": image_index,
                                "image_width": img_rect.width,
                                "image_height": img_rect.height,
                            }
                        )
                        image_index += 1

    return text, images


def extract_text_and_images_from_pdfs(
        pdfs_path: List[str],
        header_height: int,
        footer_height: int,
        min_image_size: int
) -> List[Dict]:
    logging.info("\nExtracting text and images from PDFs...")
    mongo_docs = []
    for pdf_path in tqdm(pdfs_path):
        text_by_outline, document_name = (
            extract_text_and_images_by_outline_excluding_headers_footers(pdf_path, header_height, footer_height)
        )

        for (title, page_num, outline_level, outline_sublevel), (text, images) in text_by_outline.items():
            mongo_doc = {
                "document_name": document_name,
                "title": title,
                "outline_level": outline_level,
                "outline_sublevel": outline_sublevel,
                "page": page_num,
                "text": text,
                "images": images,
            }
            mongo_docs.append(mongo_doc)

    for mongo_doc in mongo_docs:
        logging.info(f"{'*' * 40}\n{mongo_doc['document_name']}\n{'*' * 40}")
        logging.info(f"Title: {mongo_doc['title']}")
        logging.info(f"Outline Level: {mongo_doc['outline_level']}")
        logging.info(f"Outline Sublevel: {mongo_doc['outline_sublevel']}")
        logging.info(f"Page: {mongo_doc['page']}")
        logging.info(f"Images: {len(mongo_doc['images'])}")
        logging.info(f"Text characters: {len(mongo_doc['text'])}")
        logging.info(f"{'-' * 40}\n\n")

    return mongo_docs


def get_mongodb_details() -> Tuple[str, str, str]:
    logging.info("\nEnter MongoDB connection details.")
    mongodb_client = input("Enter the MongoDB host and port: ")
    mongodb_db = input("Enter the MongoDB database: ")
    mongodb_collection = input("Enter the MongoDB collection: ")
    return mongodb_client, mongodb_db, mongodb_collection


def get_mongodb_connection(
        mongodb_client: str,
        mongodb_db: str,
        mongodb_collection: str
) -> Union[Tuple[MongoClient, Database, Collection], None]:
    try:
        mongodb_client = MongoClient(mongodb_client, serverSelectionTimeoutMS=5000)
        mongodb_client.server_info()
        mongodb_db = mongodb_client[mongodb_db]
        mongodb_collection = mongodb_db[mongodb_collection]
        logging.info("\nConnected to MongoDB successfully.")
        return mongodb_client, mongodb_db, mongodb_collection
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        logging.info("")
        if ask_retry():
            return get_mongodb_connection(*get_mongodb_details())
        else:
            sys.exit(1)


def ask_retry() -> bool:
    while True:
        user_input = input(
            "\nOperation failed. Would you like to try again? [Y/n]: "
        ).lower()
        print("")
        if user_input in VALID_NEGATIVE_INPUTS:
            return False
        if user_input in VALID_AFFIRMATIVE_INPUTS:
            return True
        else:
            logging.info("Sorry, I didn't understand your input. Please try again.")


def save_text_and_images_to_mongodb(
        mongodb_collection: Collection,
        mongo_docs: List[Dict]
) -> None:
    try:
        bulk_operations = []
        for doc in mongo_docs:
            bulk_operations.append(
                UpdateOne(
                    {"document_name": doc["document_name"], "title": doc["title"], "page": doc["page"]},
                    {"$set": doc},
                    upsert=True
                )
            )
        result = mongodb_collection.bulk_write(bulk_operations)
        logging.info(
            f"\nSuccessfully upserted {result.upserted_count} documents and modified {result.modified_count} documents in MongoDB.")
    except Exception as e:
        logging.error(f"Failed to save data to MongoDB collection: {e}")
        print("")
        if ask_retry():
            client, db, collection = get_mongodb_connection(*get_mongodb_details())
            save_text_and_images_to_mongodb(collection, mongo_docs)
        else:
            sys.exit(1)


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser(description="Extract text and images from PDFs and save to MongoDB.")
    parser.add_argument("--header-height", type=int, default=50, help="Height of the header to exclude (default: 50)")
    parser.add_argument("--footer-height", type=int, default=50, help="Height of the footer to exclude (default: 50)")
    parser.add_argument("--min-image-size", type=int, default=25,
                        help="Minimum size for images to be extracted (default: 25)")
    return parser.parse_args()


if __name__ == "__main__":
    initialize_logger()
    args = parse_arguments()
    print_description()
    request_continuation()
    files = get_pdfs_file_path()
    request_continuation()
    text_and_images_from_pdfs = extract_text_and_images_from_pdfs(files, args.header_height, args.footer_height,
                                                                  args.min_image_size)
    request_continuation()
    client, db, collection = get_mongodb_connection(*get_mongodb_details())
    save_text_and_images_to_mongodb(collection, text_and_images_from_pdfs)
    client.close()
    logging.info("\nOperation completed successfully.")
    sys.exit()
