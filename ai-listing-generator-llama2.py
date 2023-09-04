# Generate listings from AI
#

import json
import streamlit as st
import boto3

# Sagemaker Endpoint deployed in us-east-1
smr = boto3.client('sagemaker-runtime', region_name="us-east-1")



with st.sidebar:
    st.header("Llama2 Parameters")
    endpoint_name = st.text_input("Endpoint Name", value="jumpstart-dft-meta-textgeneration-llama-2-13b-f")
    reponse_length = st.slider("Response Length", min_value=1, max_value=3072, value=512)
    temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)
    topP = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.9)
    st.divider()
    generate_listing = st.button("Generate Listing")


st.header("Generative AI Listings - Llama2", divider=True)

prompt = st.text_area("Listing Generation Prompt", height=150, value="Generate a real estate listing for a modern Hampton style two storey house with these features: designer kitchen with Smeg appliances , double lockup garage, fully ducted air conditioning , 4 bedrooms, large master bedroom with ensuite and walk in robe, close to shops, transports and walking distance to primary school. ")

if generate_listing:
    st.write("Generating Listing...")


    payload = {
        "inputs": [[
            {"role": "system", "content": "You are a help real estate agent, generating listings based on the provided content."},
            {"role": "user", "content": prompt}
        ]],
        "parameters": {"max_new_tokens": reponse_length, "top_p": topP, "temperature": temp}
    }

    response = smr.invoke_endpoint(EndpointName=endpoint_name, 
                                   Body=bytes(json.dumps(payload), 'utf-8'), 
                                   ContentType="application/json", 
                                   Accept="application/json", 
                                   CustomAttributes='accept_eula=true')
    response = json.loads(response['Body'].read()) 

    st.write(response[0]["generation"]["content"])