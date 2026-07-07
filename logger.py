import logging
import os


LOG_FOLDER = "logs"

os.makedirs(LOG_FOLDER, exist_ok=True)


logging.basicConfig(

    filename=os.path.join(LOG_FOLDER, "project.log"),

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)


logger = logging.getLogger(__name__)


def log_query(question):

    logger.info(f"QUESTION : {question}")


def log_answer(answer):

    logger.info(f"ANSWER : {answer}")


def log_similarity(score):

    logger.info(f"SIMILARITY : {score}")


def log_status(status):

    logger.info(f"STATUS : {status}")