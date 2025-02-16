
import pandas as pd
import gradio as gr
import joblib

le=joblib.load('le_col.pkl')
std=joblib.load('std_col.pkl')
lr=joblib.load('model.pkl')


le_col=['Location']
std_col=['Size (sqft)', 'Bedrooms', 'Bathrooms', 'Year Built','Condition']




def Predict_house_price(Location,Size,Bedrooms,Bathrooms,Yearbuilt,Condition):
    input_data=pd.DataFrame({
        'Location':[Location],
        'Size (sqft)':[Size],
        'Bedrooms':[Bedrooms],
        'Bathrooms':[Bathrooms],
        'Year Built':[Yearbuilt],
        'Condition':[Condition]
    })
    for col in le_col:
        input_data[col]=le[col].transform(input_data[col])
    input_data[std_col]=std.transform(input_data[std_col])
    prediction=lr.predict(input_data)
    return prediction[0]
    # return f"Predicted House Preice: ${prediction[0]:,.2f}"
gr.Interface(
        fn=Predict_house_price,
    inputs=[
        gr.Dropdown(
            ["Suburban","Urban","Rural"],label="Location"),
        gr.Number(label="Size (sqft)"),
        gr.Number(label="Bedrooms"),
        gr.Number(label="Bathrooms"),
        gr.Number(label="Year Built"),
        gr.Number(label="Condition")
    ],
    outputs=gr.Textbox(label='prediction'),
    title='Prediction Housin Price'
).launch()