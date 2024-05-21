import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio as asyncio
import random
from tqdm import tqdm
import instructor
import openai
from pydantic import BaseModel, Field
from asyncio import run
from itertools import product, batched
import braintrust
import hashlib


client = instructor.from_openai(openai.AsyncOpenAI())


def calculate_mrr(chunk_id, predictions):
    return 0 if chunk_id not in predictions else 1 / (predictions.index(chunk_id) + 1)


def calculate_recall(chunk_id, predictions):
    return 1 if chunk_id in predictions else 0


eval_functions = {"mrr": calculate_mrr, "recall": calculate_recall}
SIZES = [3, 5, 10, 20]


class QuestionAnswerPair(BaseModel):
    """
    This model represents a pair of a question generated from a text chunk, its corresponding answer,
    and the chain of thought leading to the answer. The chain of thought provides insight into how the answer
    was derived from the question.
    """

    chain_of_thought: str = Field(
        ..., description="The reasoning process leading to the answer."
    )
    question: str = Field(
        ..., description="The generated question from the text chunk."
    )
    answer: str = Field(..., description="The answer to the generated question.")


def create_db_if_not_exists() -> lancedb.DBConnection:
    return lancedb.connect("./db")


def create_table_if_not_exists(db: lancedb.DBConnection) -> lancedb.table.Table:
    current_tables = db.table_names()
    print(f"Current tables in the database: {current_tables}")
    if "chunk" in current_tables:
        print("Table 'chunk' already exists. Opening the existing table.")
        return db.open_table("chunk")

    print("Table 'chunk' does not exist. Creating a new table.")
    db = lancedb.connect("./db")
    func = get_registry().get("openai").create(name="text-embedding-3-small", dim=256)

    class TextChunk(LanceModel):
        chunk_id: str
        text: str = func.SourceField()
        vector: Vector(func.ndims()) = func.VectorField()

    print("Creating table 'chunk' with schema TextChunk.")
    return db.create_table("chunk", schema=TextChunk, mode="overwrite")


def get_data_and_selected_passages_from_dataset():
    dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True).take(100)

    passages = set()
    data = []
    labels = []
    for row in tqdm(dataset):
        selected_passages = []
        for idx, passage in enumerate(row["passages"]["passage_text"]):
            if passage not in passages:
                chunk_id = hashlib.md5(passage.encode()).hexdigest()
                passage_data_obj = {"text": passage, "chunk_id": chunk_id}
                data.append(passage_data_obj)
                passages.add(passage)

                if row["passages"]["is_selected"][idx]:
                    selected_passages.append(passage_data_obj)

        labels.extend(selected_passages)

    return data, labels


def insert_data_into_table_if_empty(data: list[dict], table: lancedb.table.Table):
    """
    If the table has existing data, then we don't insert any data into the table
    """
    print("Checking if the table has existing data...")
    if table.count_rows() > 0:
        print("Table already has data. No insertion needed.")
        return

    print("Table is empty. Inserting data...")
    for passage_batch in tqdm(batched(data, 20)):
        print(f"Inserting batch of size {len(passage_batch)} into the table.")
        table.add(list(passage_batch))
    print("Data insertion complete.")


async def generate_question_batch(text_chunk_batch):
    async def generate_question(text: str, chunk_id: str):
        question = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a world class AI that excels at generating hypothethical search queries. You're about to be given a text snippet and asked to generate a search query which is specific to the specific text chunk that you'll be given. Make sure to use information from the text chunk.",
                },
                {"role": "user", "content": f"Here is the text chunk : {text}"},
            ],
            response_model=QuestionAnswerPair,
            max_retries=3,
        )
        return (question, chunk_id)

    coros = [
        generate_question(item["text"], item["chunk_id"]) for item in text_chunk_batch
    ]
    res = await asyncio.gather(*coros)
    return [{"input": item[0].question, "expected": item[1]} for item in res]


def retrieve_k_relevant_chunk(input: str, search_type: str):
    db = lancedb.connect("./db")
    table = db.open_table("chunk")
    return [
        item["chunk_id"]
        for item in table.search(input, query_type=search_type)
        .limit(max(SIZES))
        .to_list()
    ]


def score(chunk_id, output):
    return {
        f"{fn_name}@{size}": eval_functions[fn_name](chunk_id, output[:size])
        for size, fn_name in product(SIZES, eval_functions.keys())
    }


if __name__ == "__main__":
    experiment = braintrust.init(project="MS-Marco-Test-Manual")
    db = create_db_if_not_exists()
    table = create_table_if_not_exists(db)

    data, labels = get_data_and_selected_passages_from_dataset()
    insert_data_into_table_if_empty(data, table)

    table.create_fts_index("text", replace=True)

    eval_data = run(generate_question_batch(labels[:10]))

    for search_type in ["fts", "hybrid"]:
        for data in eval_data:
            retrieved_chunks = retrieve_k_relevant_chunk(data["input"], search_type)
            experiment.log(
                input=data["input"],
                output=retrieved_chunks,
                expected=data["expected"],
                scores=score(data["expected"], retrieved_chunks),
                metadata={
                    "search_type": search_type,
                    "category": random.choice(
                        [
                            "business",
                            "entertainment",
                            "health",
                            "science",
                            "sports",
                            "technology",
                        ]
                    ),
                },
            )

    experiment.summarize(summarize_scores=True)
