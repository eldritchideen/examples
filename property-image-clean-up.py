# 

import streamlit as st
import boto3

rekognition = boto3.client("rekognition")


REKOGNITION_CATEGORIES = [
    "Actions",
    "Animals and Pets",
    "Apparel and Accessories",
    "Beauty and Personal Care",
    "Buildings and Architecture",
    "Colors and Visual Composition",
    "Damage Detection",
    "Education",
    "Events and Attractions",
    "Everyday Objects",
    "Expressions and Emotions",
    "Food and Beverage",
    "Furniture and Furnishings",
    "Health and Fitness",
    "Hobbies and Interests",
    "Home and Indoors",
    "Home Appliances",
    "Kitchen and Dining",
    "Materials",
    "Medical",
    "Nature and Outdoors",
    "Offices and Workspaces",
    "Patterns and Shapes",
    "Performing Arts",
    "Person Description",
    "Plants and Flowers",
    "Popular Landmarks",
    "Profession",
    "Public Safety",
    "Religion",
    "Sports",
    "Symbols and Flags",
    "Technology and Computing",
    "Text and Documents",
    "Tools and Machinery",
    "Toys and Gaming",
    "Transport and Logistics",
    "Travel and Adventure",
    "Vehicles and Automotive",
    "Weapons and Military",
]


with st.sidebar:
    confidence = st.slider("Confidence", 0, 100, 75)
    max_labels = st.number_input("Max Labels Returned", min_value=1, max_value=30, value=15)
    included_categories = st.multiselect("Include Categories", options=REKOGNITION_CATEGORIES, default=["Home and Indoors"])
    st.divider()
    detect_lables = st.button("Detect Labels")


st.header("Property Image Clean Up", divider=True)

image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if image_file is not None:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

if detect_lables:
    st.write("Detecting labels...")

    settings = {
        "GeneralLabels": {}
    }

    if included_categories:
        settings["GeneralLabels"]["LabelCategoryInclusionFilters"] = included_categories

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