## RAFT

RAFT is a recipe to adapting LLMs to domain-specific RAG. You can learn more in the release-blogs [here](https://gorilla.cs.berkeley.edu/blogs/9_raft.html) and [here](https://techcommunity.microsoft.com/t5/ai-ai-platform-blog/bg-p/AIPlatformBlog). RAFT takes an input document from the user and creates a dataset using the document, consisting of synthetically generated `{ question, answer, documents }` triplets. The dataset can then be used to fine-tune models for improved question-answering and retrieval. 

This implementation of the RAFT technique has been made for a MSc project at the Universidad de Sevilla. The main goal is to generate a dataset for training a LLM so it can provide context to the search app of a company.

The input data from the user can be either a general text document (pdf, json, or txt) for specific QA adapted to look like search-based queries. The output dataset can be in the HuggingFace format, completion format, or chat format.

## Install Dependencies

Dependencies can be installed using the following command: 

```bash
pip install -r requirements.txt
```

Arguments:
- `--datapath` - the path at which the document is located
- `--output` - the path at which to save the dataset
- `--output-format` - the format of the output dataset. Defaults to `hf` for HuggingFace. Can be one of `hf`, `completion`, `chat`.
- `--output-type` - the type of the output dataset file. Defaults to `jsonl`. Can be one of `jsonl`, `parquet`.
- `--output-chat-system-prompt` - The system prompt to use when the output format is `chat`. Optional.
- `--distractors` - the number of distractor documents to include per data point / triplet
- `--doctype` - the type of the document, must be one of the accepted doctypes
  - currently accepted doctypes: `pdf`, `txt`, `json`
  - documents in `json` format must have a "text" attribute containing the content from which chunks are extracted
- `--p` - the percentage of including the oracle documents in the context
- `--chunk_size` - the size of each chunk in number of tokens
- `--questions` - the number of data points / triplets to generate per chunk. (Note: if this parameter is not provided, the script will generate one question/input from every atomic claim made for a chunk.)
- `--octoai_key` - your OctoAI key used to make queries to their models.
- `--embedding-model` - The embedding model to use to encode documents chunks. Defaults to `thenlper/gte-large`.
- `--completion-model` - The model to use to generate answers. Defaults to `meta-llama-3-70b-instruct`.
- `--fast` - Fast mode flag. By default, this flag is not included and the script runs in safe mode, where it saves checkpoint datasets, allowing the script to recover and continue where it left off in the case of an interruption. Include this flag to run RAFT without recovery. 
-- `topic` - the topic of the document, used for the chat format. Defaults to `some topic(s)`.
-- `claim_model` - the model to use to generate claims. Defaults to `meta-llama-3-70b-instruct`.
-- `instruction_model` - the model to use to generate instructions (inputs/questions). Defaults to `meta-llama-3-70b-instruct`.
-- `perturbation_model` - the model to use to generate perturbations for the questions. If not provided, this task will be skipped.
-- `real` - Real mode flag. By default, this flag is not included and the script runs without without using a RAG to build the evaluatiion dataset. If you include this flag, you must also modify the `raft.py` script to adapt the specific function associated.

**Note**: In the `raft.py` script, you could modify the constant `N`to change the number of the chunks needed to make a checkpoint. The default value is 15.

## Usage

### Usage with OctoAI API

Run the following command with your desired arguments to generate the dataset.  
```bash 
python3 raft.py \
  --datapath PATH_TO_DATA \
  --output OUTPUT_PATH \
  --output-format hf \ # or completion or chat
  --distractors 3 \
  --p 1.0 \
  --doctype pdf \
  --chunk_size 512 \
  --octoai_key YOUR_OCTOAI_KEY
```

**Note**: You must implement the `read_rag_data` function in `raft.py` to use the RAG to build the evaluation dataset:
  ```python
  def read_rag_data(ins:str, chunk:str, chunks: 'list[str]', num_distract: int, i: int) -> str:
    """
    Generate real instructions for testing eval purposes.
    """
    # Implement this function to read the RAG data
    raise NotImplementedError
  ```

**Note**: The evaluation dataset generated is not a good representation of a real evaluation dataset. It is only used for testing purposes. You should remove all the code that generates that dataset as it is not useful for the final dataset.

**Note**: You must also create a file `.env` in the root directory with the following content (replace `<replace_me>` with your OpenAI API key):

```
# OpenAI
OCTOAI_API_KEY=<replace_me>
OCTOAI_API_TOKEN=<replace_me>
OPENAI_BASE_URL="https://text.octoai.run/v1"
```

`raft.py` does the following:  
- Takes a document located at `PATH_TO_DATA`, breaks it into chunks of size `chunk_size` tokens if the data is a pdf, json, or txt, or chunks of one API endpoint if the data is an API documentation, as denoted by `doctype`.
- For each chunk, uses a LLM to synthetically generate `questions` question-answer pairs (the questions do not resemple question phrases but typical search-based queries) and adds `distractors` distractor chunks to each pair, creating {Q, A, D} triplets. Each triplet represents one datapoint in the dataset, where Q is the question/use-case, A is the answer, and D is the relevant chunk + distractor chunks. 
- Each data point / triplet also contains other attributes (e.g. metadata), such as `id`, `type`, and `cot_answer`.
- Uses the HuggingFace Dataset API to create a dataset from all triplets and saves it at `OUTPUT_PATH` in the .arrow and .jsonl formats.

### Example Usage

This details the command and process used to generate the example dataset found in `./sample_ds4`. The document is a pdf of the Wikipedia page on the United States of America. 
```bash 
python3 raft.py --datapath sample_data/United_States_PDF.pdf --output ./sample_ds4 --distractors 4 --doctype pdf --chunk_size 512 --openai_key OPENAI_KEY
```

#### 1. Chunk generation
RAFT takes pdf and divides text into chunks of size 512 tokens. A sample chunk:  
 ```python
 "[CLS] United States of America Flag Coat of arms Motto : \" In God We Trust \" [ 1 ] Other traditional mottos : [ 2 ] \" E pluribus unum \" ( Latin ) \" Out of many, one \" \" Annuit cœptis \" ( Latin ) \" Providence favors our undertakings \" \" Novus ordo seclorum \" ( Latin ) \" New order of the ages \" Anthem : \" The Star - Spangled Banner \" [ 3 ] United States The United States of America ( USA or U. S. A. ), commonly know n as the United States ( US or U. S. ) or America, is a country primarily located in North America, between Canada and Mexico. It is a liberal democracy and republic of 50 federated states, a federal capital district ( Washington, D. C. ), and 326 Indian reservations that overlap with state boundaries. Outside the union of states, it asserts sovereignty over five major unincorporated island territories and various uninhabited islands. [ i ] The country has the world\'s third - largest land area, [ c ] largest maritime exclusive economic zone, and the third - largest population ( over 334 million ). [ j ] The federal government uses a presidential system with three separate branches : legislative, executive, and judicial. American territory was first settled by Paleo - Indians who migrated across the Bering land bridge over 12, 000 years ago. Colonization by the British began in 1607. Thirteen colonies eventually rebelled against the British Crown over taxation and political representation, declaring independence on July 4, 1776. Their victory in the American Revolutionary War ( 1775 – 83 ) resulted in a confederation of states before the U. S. Constitution and Bill of Rights were ratified. The young nation continued to acquire neighbor ing territories and spanned North America by the late 1840s. Longstanding disagreements over slavery led to the secession of the southern Confederate States of America, which were defeated by the remaining Union in the American Civil War ( 1861 – 65 ). Slavery was abolished, but discriminatory laws persisted in the South. By 1900, rapid industrialization established the United States as a great power and the world\'s largest economy. Following the Japanese attack on Pearl Harbor in December 1941, the United States joined the Allies of World War II. After their victory, it competed against the Soviet Union for dominance in nuclear and conventional"
  ```

#### 2. Question and answer generation
RAFT then uses OctoAI's models to generate all questions per chunk (based on the atomic claims previously generated) as well as the label (answer) for each question. Proceeding with the previous example chunk:  

**Questions:**  

```python
['official motto of the United States of America',
  'states in the United States of America',
  'which territories United States claim sovereignty over, outside union of states',
  'date thirteen colonies declare independence',
  'secession southern Confederate States of America']
 ```

 **Answers:**
```python
['"In God We Trust"',
 '50 federated states',
 'Five major unincorporated island territories.',
 'July 4, 1776',
 'Disagreements over slavery']
 ```
#### 3. Append distractor documents
For each question-answer pair, append 4 randomly selected chunks as distractor documents to form the {Q, A, D} triplet. Proceeding with the current example, a {Q, A, D} triplet, or one datapoint, would look like: 

```python
{
  'id': 'seed_task_0', 
  'type': 'general', 
  'question': 'What is the official motto of the United States of America?', 
  'context': {
    'sentences': [
      ["the Gulf of Mexico are prone to hurricanes, ... and enforces the Act. [ 189 ] As of 2022, the U. S",
    "energy from fossil fuel and the largest ... there are 19, 969 airports in the U. S., of which 5, 193 are designated",
    'weaponry, ideology, and international i... and is a permanent member of the UN Security Council. The first documentary evidence of the phrase " United States',
    '[CLS] United States of America Flag Coat of arms ... dominance in nuclear and conventional',
    '##om ic soft pow er. [ 405 ] [ 406 ] Nearly all present ... rights in the United States are advanced by global standards.']
    ],
    'title': [
      ['placeholder_title',
      'placeholder_title',
      'placeholder_title',
      'placeholder_title',
      'placeholder_title']
    ]
  },
  'answer': '"In God We Trust"',
  'cot_answer': None
}

```

#### 4. Generate and save dataset
RAFT repeats steps 2 and 3 for each chunk and saves the dataset to the path specified by the `--output` argument.


#### 5. Convert the dataset to the format expected for fine tuning

If you specified the `--output-format completion` or `--output-format chat` argument for the `raft.py` script, you can skip this part.

Otherwise, you need to convert the dataset to the format expected for fine-tuning a `completion` model in Azure with the following command:

```
python3 format.py --input output/data-00000-of-00001.arrow --output output.completion.jsonl --output-format completion
```

**Note**: the `format.py` script also has its own help

```
python3 format.py --help
```

**Note**: If fine tuning a chat model, then you need to use `--output-format chat` and optionally add the `--output-chat-system-prompt` parameter to configure the system prompt included in the dataset.

#### 6. Finetune your own model on Microsoft AI Studio
Once the dataset is prepared, follow the instructions in [azure-ai-studio-ft/howto.md](azure-ai-studio-ft/howto.md) to finetune and deploy your own RAFT model. Make sure to use `prompt` as input and `completion` as output when fine tuning a `completion` model and the `messages` column as input when fine tuning a `chat` model.

#### 7. Evaluate RAFT model
After deploying your model in AI Studio, use command to evaluate the RAFT model. Make sure to fill in `base_url` and `api_key` in the `eval.py`, these can be found in the AI Studio. 
```bash 
python3 eval.py --question-file YOUR_EVAL_FILE.jsonl --answer-file YOUR_ANSWER_FILE
```

The `YOUR_EVAL_FILE.jsonl` is in the format where
```python
{
  'instruction': '<DOCUMENT> document1 </DOCUMENT>\n<DOCUMENT> document2 </DOCUMENT> ...\n{question}',
  'gold_answer': '{answer}'
}
```

The answer file generated by the model is in the format where
```python
{
  'instruction': '<DOCUMENT> document1 </DOCUMENT>\n<DOCUMENT> document2 </DOCUMENT> ...\n{question}',
  'gold_answer': '{answer}',
  'model_answer': '{answer}'
}
```
### Disclaimer
This project is based on the RAFT project by the University of California, Berkeley. The original project can be found [here](https://github.com/ShishirPatil/gorilla). The original project is licensed under the Apache License 2.0.

'''
The original copyright notice is left below:
Copyright 2024 Tianjun Zhang and Shishir G. Patil and Naman Jain and Sheng Shen and Matei Zaharia and Ion Stoica and Joseph E. Gonzalez

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at [url](https://www.apache.org/licenses/LICENSE-2.0).
'''