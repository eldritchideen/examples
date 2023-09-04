# Classify your image anytime anywhere
#


import streamlit as st
import boto3

rekognition = boto3.client("rekognition")


# Subset of all labels we want to detect in images. 
# Full list of all labels can be found here:
#     https://docs.aws.amazon.com/rekognition/latest/dg/samples/AmazonRekognitionLabels_v3.0.zip
REKOGNITION_LABELS  = [
    "Kitchen",
    "Living Room",
    "Room",
    "Bedroom",
    "Bathroom",
    "Billiard Room",
    "Dining Room",
    "Office",
    "Dressing Room",
    "Toilet",
    "Garden",
    "Backyard",
    "Pool",
    "Garage",
    "Yard",
    "Portico",
    "Hallway",
    "Toolshed",
    "Deck",
    "Patio",
    "Laundry",
    "Balcony"
]


with st.sidebar:
    confidence = st.slider("Confidence", 0, 100, 75)
    max_labels = st.number_input("Max Labels Returned", min_value=1, max_value=30, value=15)
    included_labels = st.multiselect("Include Labels", options=REKOGNITION_LABELS)
    st.divider()
    detect_lables = st.button("Detect Labels")


st.header("Classify Your Image Anytime Anywhere", divider=True)

image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if image_file is not None:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

if detect_lables:
    st.write("Detecting labels...")

    settings = {
        "GeneralLabels": {}
    }

    if included_labels:
        settings["GeneralLabels"]["LabelInclusionFilters"] = included_labels
    else: 
        # If no labels are selected, detect all labels
        settings["GeneralLabels"]["LabelInclusionFilters"] = REKOGNITION_LABELS

    response = rekognition.detect_labels(
        Image={
            "Bytes": image_file.getvalue()
        }, 
        MaxLabels=max_labels,
        MinConfidence=confidence,
        Features=["GENERAL_LABELS"],
        Settings=settings,
    )
    
    st.write(response["Labels"])