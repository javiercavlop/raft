import pickle
from mdc import MDC
from logconf import log_setup
import logging
from typing import Literal, Any
import argparse
from octoai.client import OctoAI
import datasets
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import json
import PyPDF2
import random
import os, shutil
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OctoAIEmbeddings
from client_utils import build_octoai_client, build_langchain_embeddings
from math import ceil
from format import DatasetConverter, datasetFormats, outputDatasetTypes
from octoai.text_gen import ChatMessage

log_setup()

logger = logging.getLogger("raft")

DocType = Literal["pdf", "json", "txt"]

# Every N chunks, save checkpoint
N = 15

def get_args() -> argparse.Namespace:
    """
    Parses and returns the arguments specified by the user's command
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--datapath", type=str, default="", help="The path at which the document is located")
    parser.add_argument("--output", type=str, default="./", help="The path at which to save the dataset")
    parser.add_argument("--output-format", type=str, default="hf", help="Format to convert the dataset to. Defaults to hf.", choices=datasetFormats)
    parser.add_argument("--output-type", type=str, default="jsonl", help="Type to export the dataset to. Defaults to jsonl.", choices=outputDatasetTypes)
    parser.add_argument("--output-chat-system-prompt", type=str, help="The system prompt to use when the output format is chat")
    parser.add_argument("--distractors", type=int, default=3, help="The number of distractor documents to include per data point / triplet")
    parser.add_argument("--questions", type=int, help="The number of data points / triplets to generate per chunk")
    parser.add_argument("--p", type=float, default=1.0, help="The percentage that the oracle document is included in the context")
    parser.add_argument("--chunk_size", type=int, default=512, help="The size of each chunk in number of tokens")
    parser.add_argument("--doctype", type=str, default="pdf", help="The type of the document, must be one of the accepted doctypes", choices=["pdf", "txt", "json"])
    parser.add_argument("--octoai_key", type=str, default=None, help="Your OctoAI key used to make queries to Llama models")
    parser.add_argument("--embedding_model", type=str, default="thenlper/gte-large", help="The embedding model to use to encode documents chunks (thenlper/gte-large, ...)")
    parser.add_argument("--completion_model", type=str, default="meta-llama-3-70b-instruct", help="The model to use to generate questions and answers (meta-llama-3-70b-instruct, mixtral-8x22b-instruct, ...)")
    parser.add_argument("--fast", action="store_true", help="Run the script in fast mode (no recovery implemented)")
    parser.add_argument("--topic", type=str, default="some topic(s)", help="The topic of the document")
    parser.add_argument("--claim_model", type=str, default="meta-llama-3-70b-claim", help="The model to use to generate atomic claims (meta-llama-3-70b-claim, mixtral-8x22b-claim, ...)")
    parser.add_argument("--instruction_model", type=str, default="meta-llama-3-70b-instruct", help="The model to use to generate instructions (meta-llama-3-70b-instruct, mixtral-8x22b-instruct, ...)")
    parser.add_argument("--perturbation_model", type=str, help="The model to use to perturbate questions (meta-llama-3-70b-instruct, mixtral-8x22b-instruct, ...)")
    parser.add_argument("--real", action="store_true", help="Generate real data for testing eval purposes")

    args = parser.parse_args()
    return args


def get_chunks(
    file_path: str, 
    doctype: DocType = "pdf", 
    chunk_size: int = 512, 
    octoai_key: str or None = None,
    model: str = None
) -> 'list[str]':
    """
    Takes in a `file_path` and `doctype`, retrieves the document, breaks it down into chunks of size
    `chunk_size`, and returns the chunks.
    """
    chunks = []

    logger.info(f"Retrieving chunks from {file_path} of type {doctype}")
    
    if doctype == "json":
        with open(file_path, 'r') as f:
            text = json.load(f)["text"]
    elif doctype == "pdf":
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
    elif doctype == "txt":
        with open(file_path, 'r') as file:
            data = file.read()
        text = str(data)
    else:
        raise TypeError("Document is not one of the accepted types: pdf, json, txt")
    
    num_chunks = ceil(len(text) / chunk_size)
    logger.info(f"Splitting text into {num_chunks} chunks using the {model} model. (It may take a while).")
    
    embeddings = build_langchain_embeddings(endpoint_url="https://text.octoai.run/v1/embeddings")
    text_splitter = SemanticChunker(embeddings, number_of_chunks=num_chunks)
    chunks = text_splitter.create_documents([text])
    chunks = [chunk.page_content for chunk in chunks]
            
    return chunks

def extract_atomic_claims(client: OctoAI, chunk: Any, t: str = "some topic(s)", model: str = None) -> 'list[str]':
    """
    Extracts the atomic claims from `chunk`. Used when the input document is of general types 
    `pdf`, `json`, or `txt`.
    """
    response = client.text_gen.create_chat_completion(
        model=model,
        messages=[
            ChatMessage(role = "system", content = f"You are a synthetic atomic claim generator. Given a chunk of context about {t}, generate all the atomic claims that can be made based on the information in the chunk. For example, if the given context was a Wikipedia paragraph about the United States, an atomic claim could be 'The United States is a country in North America.'"),
            ChatMessage(role = "system", content = "The claims should be short and to the point. The claims must be unique, so ensure there are no repetitions. Include only the claims in your response (do not add any other piece of information or clarification; write just the claims). If there are no relevant atomic claims, you should respond with 'No atomic claims found.'"),
            ChatMessage(role = "user", content = str(chunk))
        ],
    )
    claims = response.choices[0].message.content.split('\n')
    claims = [strip_str(q) for q in claims]
    claims = [q for q in claims if any(c.isalpha() for c in q)]
    claims = [q.strip() for q in claims if q and q.strip() and "claim" not in q.lower() and len(q.split(" ")) > 3]

    return claims

def generate_instructions(client: OctoAI, chunk: Any, claim: str, t: str = "some topic(s)", model: str = None) -> 'dict[list[str]]':
    """
    Generates a question (based in the claim passed) for `chunk`. Used when the input document is of general types 
    `pdf`, `json`, or `txt`.
    """
    response = client.text_gen.create_chat_completion(
        model=model,
        messages=[
            ChatMessage(role = "system", content = f"You are a synthetic input-answer pair generator. Given a chunk of context about {t}, generate one example input a user could ask that would be answered using information from the atomic claim. The inputs must resemble typical search engine queries, potentially non-question phrases, and should not be excessively long (usually these queries may follow a concatenate keywords query such as 'year foundation USA' instead of 'Which whas the year of the foundation of the USA?'). The inputs should solicit a response that provides context and elaboration based on the atomic claim. For example, if the given context is a Wikipedia paragraph about the United States and the atomic claim is 'The United States of America is a federal republic consisting of 50 states,' an example input could be 'states in the United States.'."),
            ChatMessage(role = "system", content = "Your output should include:\nThe user input (formatted as a typical search query and written between the block <INPUT></INPUT>).\nA corresponding answer that elaborates on the atomic claim to provide useful context for the user (it must be written between the block <GOLD></GOLD>). Each part of the answer must be contained in the same line. Remember to follow the format specified, especially when generating the search query based input so it resembles non-questions phrases."),
            ChatMessage(role = "user", content = (f"Chunk: '{str(chunk)}'\nAtomic claim: '{claim}'"))
        ],
        
    )
    queries = response.choices[0].message.content.split('\n')
    queries = [strip_str(q) for q in queries]
    queries = [q for q in queries if any(c.isalpha() for c in q)]

    inputs = [i.replace("</INPUT>", "").replace("INPUT>", "").strip() for i in queries if "INPUT" in i]
    golden = [i.replace("</GOLD>", "").replace("GOLD>", "").strip() for i in queries if "GOLD" in i]

    inputs = [i for i in inputs if i]
    golden = [i for i in golden if i]
    
    return {"inputs": inputs, "gold": golden} 

def generate_instructions_by_questions(client: OctoAI, chunk: Any, n: int, t: str = "some topic(s)", model: str = None) -> 'dict[list[str]]':
    """
    Generates n inputs for `chunk`. Used when the input document is of general types 
    `pdf`, `json`, or `txt`.
    """
    response = client.text_gen.create_chat_completion(
        model=model,
        messages=[
            ChatMessage(role = "system", content = f"You are a synthetic input-answer pair generator. Given a chunk of context about {t}, generate {n} example inputs a user could ask that would be answered using information from the chunk. The inputs must resemble typical search engine queries, potentially non-question phrases, and should not be excessively long (usually these queries may follow a concatenate keywords query such as 'year foundation USA' instead of 'Which whas the year of the foundation of the USA?')."),
            ChatMessage(role = "system", content = "Include only the inputs generated in your response."),
            ChatMessage(role = "user", content = (f"{str(chunk)}"))
        ],
        
    )
    queries = response.choices[0].message.content.split('\n')
    queries = [strip_str(q) for q in queries]
    queries = [q for q in queries if any(c.isalpha() for c in q)]
    queries = [q.strip() for q in queries if q and q.strip() and not (q.strip().lower().startswith("here") and ("example" in q.strip().lower() or "input" in q.strip().lower()))]
    
    return queries

def strip_str(s: str) -> str:
    """
    Helper function for helping format strings returned by the model.
    """
    l, r = 0, len(s)-1
    beg_found = False
    for i in range(len(s)):
        if s[i].isalpha():
            if not beg_found:
                l = i
                beg_found = True
            else:
                r = i 
    r += 2
    return s[l:min(r, len(s))]

def encode_question_gen(question: str, chunk: Any, t: str = "some topic(s)") -> 'list[str]':
    """
    Encode multiple prompt instructions into a single string for the general case (`pdf`, `json`, or `txt`).
    """
    
    prompts = []
        
    prompt = """
        Question: {question}\nContext: {context}\n
        Answer this question using the information given in the context above. Take into account that the question may resemble typical search engine queries (potentially non-question phrases). Here is things to pay attention to: 
        - First provide step-by-step reasoning on how to answer the question. 
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context. 
        - End your response with final answer in the form <ANSWER>: $answer, the answer should be informative and elaborated as it will add valuable insights to the user making the search.
        You MUST begin your final answer with the tag "<ANSWER>:".
    """.format(question=question, context=str(chunk))
    prompts.append(ChatMessage(role = "system", content = f"You are a helpful question answerer who can provide a detailed answer given a question (or search input) and relevant context (about {t})."))
    prompts.append(ChatMessage(role = "user", content = prompt))
    return prompts

def generate_label(client: OctoAI, question: str, context: Any, doctype: DocType, t: str = "some topic(s)", model: str = None) -> str or None:
    """
    Generates the label / answer to `question` using `context`
    """
    question = encode_question_gen(question, context, t)
    response = client.text_gen.create_chat_completion(
        model=model,
        messages=question,
        n=1,
        temperature=0
    )
    response = response.choices[0].message.content
    return response

def perturbate_question(client: OctoAI, ins: str, model: str = None) -> 'list[str]':
    """
    Generates perturbated versions of the input `ins`.
    """
    response = client.text_gen.create_chat_completion(
        model=model,
        messages=[
            ChatMessage(role = "system", content = "You are an assistant for rephrasing and perturbating inputs. You must take the input provided by the user. This input will be a search query, which may resemble keyword-based search phrases and may not always be in question form. You have to create all the possible rephrased versions of the input without changing its context or meaning. You must ensure the outputs are diverse and free of repetitions. You have to maintain the integrity of the search intent behind the input."),
            ChatMessage(role = "system", content = "You must ensure that each output must be a rephrased or perturbed version of the input query. Remember no duplicate outputs are allowed. If unable to generate rephrased versions, respond with: 'Sorry, I cannot do the task'. Please do not add any additional information or clarification to the answer provided."),
            ChatMessage(role = "system", content = "The following example input is provided to help you understand the task: 'best places to visit in Europe summer'. A good answer of this input could be: 'top destinations in Europe for summer travel\npopular European summer vacation spots\nbest European cities to visit during summer\nEurope summer travel guide\ntop summer holiday destinations in Europe'."),
            ChatMessage(role = "user", content = ins)
            ]
    )
    queries = response.choices[0].message.content.split('\n')
    queries = [strip_str(q) for q in queries]
    queries = [q for q in queries if any(c.isalpha() for c in q)]
    queries = [q.strip() for q in queries if q and q.strip() and not (q.strip().lower().startswith("here") and ("rephrased" in q.strip().lower() or "input" in q.strip().lower())) and q.strip().lower() != "sorry, i cannot do the task"]
    
    return queries

def add_chunk_to_dataset(
    client: OctoAI,
    chunks: 'list[str]', 
    chunk: str, 
    doctype: DocType,
    perturbation_model: str,
    num_questions: int,
    real: bool = False,
    t: str = "some topic(s)",
    num_distract: int = 3, 
    p: float = 0.8,
    claimModel: str = None,
    instructionModel: str = None,
    model: str = None
) -> 'list[dict[str]]':
    """
    Given a chunk, create {Q, A, D} triplets and add them to the dataset.
    """
    global ds
    i = chunks.index(chunk)
    
    acs = extract_atomic_claims(client, chunk, model=claimModel, t=t)
    
    if not acs:
        logger.info(f"Extracted {len(acs)} atomic claims from the chunk. The model used was {claimModel}.")
        logger.info(f"Skipping chunk {i} as no atomic claims were found.")
        if not num_questions:
            return []
    
    qs = []
    gs = []
    eval_dataset = []
    
    if acs:
        logger.info(f"Extracted {len(acs)} atomic claims from the chunk. The model used was {claimModel}.")
        if len(acs) > 0:
            logger.info(f"Example claim: {acs[0]}")
        
        for claim in acs:
            fqs = generate_instructions(client, chunk, claim, t, model=instructionModel)
            qs += fqs['inputs']
            gs += fqs['gold']
        
        eval_qs = zip(qs, gs)
        eval_dataset = []
    
        logger.info(f"Adding {len(qs)} questions to the dataset. The model used was {instructionModel}.")
        if len(qs) > 0:
            logger.info(f"Example question: {qs[0]}")
        
        for ins, gld in eval_qs:
            dataln = {
                "instruction": None,
                "gold_answer": None
            }
            
            data_context = ""
            if not real:
                docs = [chunk]
                
                indices = list(range(0, len(chunks)))
                indices.remove(i)
                for j in random.sample(indices, num_distract):
                    docs.append(chunks[j])
                random.shuffle(docs)
                for doc in docs:
                    data_context += "<DOCUMENT>" + str(doc) + "</DOCUMENT>\n"
            else:
                data_context += generate_instruction_ais_channel(ins, chunk, chunks, num_distract, i)
            data_context += ins
            dataln["instruction"] = data_context
            dataln["gold_answer"] = gld
            eval_dataset.append(dataln)
        
        logger.info("Added the questions and gold answers to the dataset.")
        if not real:
            logger.info("Example fake question and gold answer:")
        else:
            logger.info("Example real question and gold answer:")
        if len(eval_dataset) > 0:
            logger.info(f"Question: {eval_dataset[0]['instruction']}")
            logger.info(f"Gold answer: {eval_dataset[0]['gold_answer']}")
    
        if perturbation_model:
            new_qs = []
            for q in qs:
                perturbated_qs = perturbate_question(client, q, model=perturbation_model)
                perturbated_qs.append(q)
                perturbated_qs = list(set(perturbated_qs))
                perturbated_qs.remove(q)
                logger.info(f"Generated {len(perturbated_qs)} perturbated questions for the question '{q}'. The model used was {perturbation_model}.")
                if len(perturbated_qs) > 0:
                    logger.info(f"Example perturbated question: {perturbated_qs[0]}")
                new_qs += perturbated_qs
            
            qs += new_qs
        
        logger.info(f"Now adding the questions to the dataset. The model that is goingo to be used is {model}. This process may take a while")
        
    if num_questions:
        qs = generate_instructions_by_questions(client, chunk, num_questions, t, model=instructionModel)
        logger.info(f"Adding {len(qs)} (should be {num_questions}) questions to the training dataset. The model used was {instructionModel}.")
        if len(qs) > 0:
            logger.info(f"Example question: {qs[0]}")
    
    for q in qs:
        datapt = {
            "id": None,
            "type": None,
            "question": None,
            "context": None,
            "oracle_context": None,
            "cot_answer": None
        }

        datapt["id"] = f"seed_task_{0 if not ds else ds.num_rows}"
        datapt["type"] = "general"
        datapt["question"] = q

        # add num_distract distractor docs
        docs = [chunk]
        indices = list(range(0, len(chunks)))
        indices.remove(i)
        for j in random.sample(indices, num_distract):
            docs.append(chunks[j])
        # decides whether to add oracle document
        oracle = random.uniform(0, 1) < p
        if not oracle:
            docs[0] = chunks[random.sample(indices, 1)[0]]
        random.shuffle(docs)

        d = {
            "title": [],
            "sentences": []
        }

        d["title"].append(["placeholder_title"]*(num_distract+1))
        d["sentences"].append(docs)
        datapt["context"] = d
        datapt["oracle_context"] = chunk

        # add answer to q
        datapt["cot_answer"] = generate_label(client, q, chunk, doctype, t=t, model=model) 

        # construct model instruction 
        context = ""
        for doc in docs:
            context += "<DOCUMENT>" + str(doc) + "</DOCUMENT>\n"
        context += q
        datapt["instruction"] = context

        # add to dataset
        if not ds:
            # init ds
            datapt["id"] = [datapt["id"]]
            datapt["type"] = [datapt["type"]]
            datapt["question"] = [datapt["question"]]
            datapt["context"] = [datapt["context"]]
            datapt["oracle_context"] = [datapt["oracle_context"]]
            datapt["cot_answer"] = [datapt["cot_answer"]]
            datapt["instruction"] = [datapt["instruction"]]
            ds = Dataset.from_dict(datapt)
        else:
            ds = ds.add_item(datapt)
    
    return eval_dataset

def read_rag_data(ins:str, chunk:str, chunks: 'list[str]', num_distract: int, i: int) -> str:
    """
    Generate real instructions for testing eval purposes.
    """
    # Implement this function to read the RAG data
    raise NotImplementedError

def save_checkpoint(state, filename):
    with open(filename, 'w') as f:
        f.write(str(state))

def load_checkpoint(filename):
    with open(filename, 'r') as f:
        return int(f.read())

def main():
    global ds

    # run code
    args = get_args()

    # Validate arguments
    if args.output_chat_system_prompt and args.output_format != "chat":
        raise Exception("Parameter --output-chat-system-prompt can only be used with --output-format chat")

    OCTOAI_API_KEY = args.octoai_key

    client = build_octoai_client(
        api_key=OCTOAI_API_KEY,
    )

    CHUNK_SIZE = args.chunk_size
    NUM_DISTRACT_DOCS = args.distractors

    chunks = get_chunks(args.datapath, args.doctype, CHUNK_SIZE, OCTOAI_API_KEY, model=args.embedding_model)

    ds = None
    eval_dataset = []

    num_chunks = len(chunks)

    if not args.fast:
        start = 0
        if os.path.exists("checkpoint.txt"):
            start = int(load_checkpoint("checkpoint.txt"))

        for i in range((start//N)*N, len(chunks)):
            chunk = chunks[i]
            save_checkpoint(i, "checkpoint.txt")

            perc = ceil(i / num_chunks * 100)
            with MDC(progress=f"{perc}%"):
                logger.info(f"Adding chunk {i}/{num_chunks}")
                eval_dataset += add_chunk_to_dataset(client, chunks, chunk, args.doctype, args.perturbation_model, args.questions, args.real, args.topic, NUM_DISTRACT_DOCS, model=args.completion_model, instructionModel=args.instruction_model, claimModel=args.claim_model, p=args.p)

            if (i+1) % N == 0 and ds:
                ds.save_to_disk(args.output + "-checkpoints-" + str(i))
                ds = None
                with open(args.output + "-eval-checkpoints.pkl", "wb") as f:
                    pickle.dump(eval_dataset, f)
    
        if ds:
            ds.save_to_disk(args.output + "-checkpoints-last")

        ds_list = []

        for filename in os.listdir(os.path.dirname(args.output)):
            if "-checkpoints-" in filename:
                for f in os.listdir(os.path.dirname(args.output) + "/" + filename):
                    if f.endswith(".arrow"):
                        ds_list.append(Dataset.from_file(os.path.dirname(args.output) + "/" + filename + "/" + f))
        
        if os.path.exists(args.output + "-eval-checkpoints.pkl"):
            with open(args.output + "-eval-checkpoints.pkl", "rb") as f:
                help_ds = eval_dataset
                recovered_ds = pickle.load(f)
                eval_dataset = recovered_ds + help_ds

        with open(args.output + "-eval-checkpoints.pkl", "wb") as f:
            pickle.dump(eval_dataset, f)

        ds = datasets.concatenate_datasets(ds_list)
    else:
        for i, chunk in enumerate(chunks):
            perc = ceil(i / num_chunks * 100)
            with MDC(progress=f"{perc}%"):
                logger.info(f"Adding chunk {i}/{num_chunks}")
                eval_dataset += add_chunk_to_dataset(client, chunks, chunk, args.doctype, args.perturbation_model, args.questions, args.real, args.topic, NUM_DISTRACT_DOCS, model=args.completion_model, instructionModel=args.instruction_model, claimModel=args.claim_model, p=args.p)
    
    # Save as .arrow format
    ds.save_to_disk(args.output)
    
    with open(args.output + "-eval.jsonl", "w") as f:
        for entry in eval_dataset:
            json.dump(entry, f)
            f.write('\n')

    # Save as .jsonl format
    formatter = DatasetConverter()

    # Extract format specific params
    format_params = {}
    if args.output_chat_system_prompt:
        format_params['system_prompt'] = args.output_chat_system_prompt

    formatter.convert(ds=ds, format=args.output_format, output_path=args.output, output_type=args.output_type, params=format_params)

    if not args.fast:
        os.remove("checkpoint.txt")
        for filename in os.listdir(os.path.dirname(args.output)):
            if "-checkpoints-" in filename:
                shutil.rmtree(os.path.dirname(args.output) + "/" + filename)

if __name__ == "__main__":
    with MDC(progress="0%"):
        main()