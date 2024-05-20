import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio as asyncio
from braintrust import Eval
from tqdm import tqdm
import instructor
import openai
from pydantic import BaseModel, Field
from asyncio import run
from itertools import product, batched


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


def setup_table():
    import os

    if os.path.exists("./db"):
        db = lancedb.connect("./db")
        return db.open_table("chunk")

    # Connect to the LanceDB database
    db = lancedb.connect("./db")
    func = get_registry().get("openai").create(name="text-embedding-3-small", dim=256)

    class TextChunk(LanceModel):
        chunk_id: str
        text: str = func.SourceField()
        vector: Vector(func.ndims()) = func.VectorField()

    table = db.create_table("chunk", schema=TextChunk, mode="overwrite")
    load_data(table)


def load_data(table):
    fw = load_dataset(
        "HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True
    ).take(100)

    row_data = []
    for row in tqdm(fw):
        text = row["text"][:2000]  # Truncate if longer than 2000 chars
        chunk_id = row["id"]
        row_data.append({"text": text, "chunk_id": chunk_id})

    batches = batched(row_data, 20)

    for batch in batches:
        table.add(list(batch))

    table.create_fts_index("text", replace=True)


def fetch_all_chunks_from_db(max_chunk=None):
    # Connect to the LanceDB database
    db = lancedb.connect("./db")

    # Open the table
    table = db.open_table("chunk")

    # Fetch all records from the table
    all_records = table.to_pandas()

    chunks = [
        [item["chunk_id"], item["text"]]
        for item in all_records[["chunk_id", "text"]].to_dict(orient="records")
    ]

    if max_chunk is not None:
        chunks = chunks[:max_chunk]

    return chunks


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

    coros = [generate_question(text, chunk_id) for chunk_id, text in text_chunk_batch]
    res = await asyncio.gather(*coros)
    return [{"input": item[0].question, "expected": item[1]} for item in res]


def retrieve_k_relevant_chunk(input: str):
    db = lancedb.connect("./db")
    table = db.open_table("chunk")
    return [
        item["chunk_id"] for item in table.search(input).limit(max(SIZES)).to_list()
    ]


def score(question, chunk_id, output):
    from braintrust import Score

    return [
        Score(
            name=f"{fn_name}@{size}",
            score=eval_functions[fn_name](chunk_id, output[:size]),
        )
        for size, fn_name in product(SIZES, eval_functions.keys())
    ]


if __name__ == "__main__":
    setup_table()
    text_chunks = fetch_all_chunks_from_db(20)
    eval_data = run(generate_question_batch(text_chunks))
    Eval(
        "Query Test",
        data=eval_data,
        task=retrieve_k_relevant_chunk,
        scores=[score],
        # trial_count=3,
    )
