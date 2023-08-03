import streamlit as st
from pickle5 import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_data():
    data=pd.read_csv('/Users/vishalroy/Downloads/devloper/breast cancer detection/data.csv')
#    print(data.head())
    data=data.drop(['Unnamed: 32','id'],axis=1) # redundant column
    # convert target column 
    data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
    return data
    

def add_sidebar():
    st.sidebar.header("Cell measurements given by lab:")
    data=get_data()
    slider_labels=[
       ("Radius (mean)", "radius_mean"),
       ("Texture (mean)", "texture_mean"),
       ("Perimeter (mean)", "perimeter_mean"),
       ("Area (mean)", "area_mean"),
       ("Smoothness (mean)", "smoothness_mean"),
       ("Compactness (mean)", "compactness_mean"),
       ("Concavity (mean)", "concavity_mean"),
       ("Concave points (mean)", "concave points_mean"),
       ("Symmetry (mean)", "symmetry_mean"),
       ("Fractal dimension (mean)", "fractal_dimension_mean"),
       ("Radius (se)", "radius_se"),
       ("Texture (se)", "texture_se"),
       ("Perimeter (se)", "perimeter_se"),
       ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    
    inputs_dict={}

    for label ,key in slider_labels:
        inputs_dict[key]=st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean()),
        )
    return inputs_dict

def get_scaled_values(inputs_dict):
    data=get_data()
    X=data.drop(['diagnosis'],axis=1)

    scaled_X={}
    for key,value in inputs_dict.items():
        min_value=X[key].min()
        max_value=X[key].max()
        scaled_X[key]=(value-min_value)/(max_value-min_value)

    return scaled_X

def make_predictions(inputs_dict):
    model=pickle.load(open('model.pkl','rb'))
    scaler=pickle.load(open('scaler.pkl','rb'))

    input_arr=np.array(list(inputs_dict.values())).reshape(1,-1)

    #scaled the input array
    scaled_input_arr=scaler.transform(input_arr)
    pred=model.predict(scaled_input_arr)
    
    st.subheader('Cell cluster type:')

    if pred[0]==0:
        st.write('[Benign]')
    else:
        st.write('[malacious]')
#     st.write(scaled_input_arr)
    st.write("probability of being Benign:",model.predict_proba(scaled_input_arr)[0][0])
    st.write("probability of being Malacious:",model.predict_proba(scaled_input_arr)[0][1])
    st.write("Desclaimer \n this app only for expert who poses domain knowledge or doctor")
def get_radar_plot(input_data):
    input_data=get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']
    
    fig =go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
     ))
    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'],
          input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
          input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'],
          input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
     ))
    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='worst value'
     ))
    
    fig.update_layout(
       polar=dict(
          radialaxis=dict(
             visible=True,
             range=[0,1]
          )
       ),
       showlegend=True
    )
    return fig


def main():
    # UI of app 
    st.set_page_config(
        page_title=("Breast Cancer Predictor"),
        page_icon=("doctor:"),
        layout="wide",
        initial_sidebar_state="expanded"
    )

    input_measure=add_sidebar()
    #st.write(input_measure)

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("this app work on the reports of lab ,to check that whether a person has breast cancer or not using machine learning. ")
    col1,col2 =st.columns([4,1])

    with col1 :
        radar_plot=get_radar_plot(input_measure)
        st.plotly_chart(radar_plot)

    with col2 :
        make_predictions(input_measure)

    

if __name__ == '__main__':
    main()
