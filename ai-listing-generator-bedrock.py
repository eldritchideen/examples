# Generate listings from AI
#

import json
import streamlit as st
import boto3

bedrock = boto3.client("bedrock", region_name="us-west-2", endpoint_url="https://prod.us-west-2.frontend.bedrock.aws.dev")


with st.sidebar:
    st.header("Bedrock Parameters")
    model = st.selectbox("Model", options=["amazon.titan-tg1-xlarge", "amazon.titan-tg1-large"])
    reponse_length = st.slider("Response Length", min_value=1, max_value=3072, value=512)
    temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)
    topP = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.9)
    st.divider()
    generate_listing = st.button("Generate Listing")


st.header("Generative AI Listings", divider=True)

prompt = st.text_area("Listing Generation Prompt", height=300, value="Generate a real estate listing for a modern Hampton style two storey house with these features: designer kitchen with Smeg appliances , double lockup garage, fully ducted air conditioning , 4 bedrooms, large master bedroom with ensuite and walk in robe, close to shops, transports and walking distance to primary school. ")

if generate_listing:
    st.write("Generating Listing...")

    body = json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": reponse_length,
                "stopSequences":[],
                "temperature": temp,
                "topP": topP
            }
        })
    modelId = model
    accept = 'application/json'
    contentType = 'application/json'
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    
    st.write(f"""
### Listing             
{response_body["results"][0]["outputText"]}
            """)

    st.divider()
    st.write(f"""
### API Stats  
- Input Token Count: {response_body["inputTextTokenCount"]}           
- Output Token Count: {response_body["results"][0]["tokenCount"]}
- Completion Reason: {response_body["results"][0]["completionReason"]}
             """)