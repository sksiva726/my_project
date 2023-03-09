
import pandas as pd
import numpy as np
import streamlit as st 
import pickle
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing # holts winters Exponential Smoothing

model = pickle.load(open('hwe_model_mul_add_final.pickle', 'rb'))
# opc_bags1 = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\sri ra\cement_hwe_2023\opc_bags1.csv")
# model = pickle.load(open('model_mul_lin.pickle', 'rb'))
from sqlalchemy import create_engine  
start = 38
end = 49

def predict(data, user, pw, db):

    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
    # data = pd.to_datetime(data['Date'])
    global start, end
    prediction = model.predict(start, end)
    
    data["forecasted_Sales"] = np.array(prediction)
    
    data.to_sql('forecast_Sales', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

    return data


def main():
    

    st.title("Cement_Forecasting")
    st.sidebar.title("Cement_Forecasting")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Forecasting </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file" ,type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
    if uploadedFile is not None :
        try:

            data=pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.sidebar.warning("you need to upload a csv or excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict(data, user, pw, db)
        #st.dataframe(result) or
        #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap=True)
        st.table(result.style.background_gradient(cmap=cm).set_precision(2))

                           
if __name__=='__main__':
    main()


