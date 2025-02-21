from preprocess_for_prediction import preprocess
import gradio as gr
import tensorflow as tf
import pandas as pd

# load model
model = tf.keras.models.load_model('./Train/mlp_model.h5')

# Get Applicants' names
def getName(path='./combined_application_data_2021-23.csv'):
    all_staff = pd.read_csv(path)
    names = all_staff['Applicant Full Name']

    return names

def predict(data_array):
    prediction = model.predict(data_array)
    return prediction

def get_results(path):
    data_array = preprocess(path)
    result = predict(data_array)
    name_column = getName()
    # Convert the result to a DataFrame
    result_df = pd.DataFrame(result, columns=["Likelihood"])

    # Add a column called "Applicant Name"
    result_df['Applicant Name'] = name_column
    
    # Save the results to a text file
    result_df.to_csv("prediction_results.csv")  
    
    return result_df.head(10)

def download_results(output_text):
    with open("results.txt", "w") as file:
        file.write(output_text)

 
# Define input and output components
input_component = gr.File(label="Upload CSV File")
output_component = gr.Textbox(label="Candidates' Likelihoods Of Acceptance (Full results will be saved to your current folder)")

if __name__ == '__main__':
    
    iface = gr.Interface(fn=get_results, 
                         inputs=input_component, 
                         outputs=output_component,
                         title="MLP Model Predictions")
    iface.launch(share=True)
 
    