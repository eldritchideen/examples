from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
import boto3
import json

sentences = [
    # Pets
    "Your dog is so cute.",
    "How cute your dog is!",
    "You have such a cute dog!",
    # Cities in the US
    "New York City is the place where I work.",
    "I work in New York City.",
    # Color
    "What color do you like the most?",
    "What is your favourite color?",
]

embeddings = BedrockEmbeddings(region_name="us-west-2", endpoint_url="https://prod.us-west-2.frontend.bedrock.aws.dev")
local_vector_store = FAISS.from_texts(sentences, embeddings)

#query = "What type of pet do I have?"
query = "Where do I live?"
docs = local_vector_store.similarity_search(query)
context = ""

for doc in docs:
    context += doc.page_content

prompt = f"""Use the following pieces of context to answer the question at the end.

{context}

Question: {query}
Answer:"""

bedrock = boto3.client("bedrock", region_name="us-west-2", endpoint_url="https://prod.us-west-2.frontend.bedrock.aws.dev")

def call_bedrock(prompt):
    prompt_config = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0.5,
            "topP": 0.2,
        },
    }

    body = json.dumps(prompt_config)

    modelId = "amazon.titan-tg1-large"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("results")[0].get("outputText")
    return results

generated_text = call_bedrock(prompt)
print(generated_text)
print(docs)