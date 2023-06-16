import streamlit as st
import pandas as pd
import joblib
import category_encoders
import sklearn
from xgboost import XGBRegressor 


Inputs = joblib.load("Inputs.pkl")
Model = joblib.load("Model.pkl")

def prediction(Airline, Source, Destination,Dep_Time, Arrival_Time,
       Total_Stops, Additional_Info, Month_of_Journey, Day_of_Journey,
       Duration_min):
    test_df = pd.DataFrame(columns=Inputs)
    test_df.at[0,"Airline"] = Airline
    test_df.at[0,"Source"] = Source
    test_df.at[0,"Destination"] = Destination
    test_df.at[0,"Dep_Time"] = Dep_Time
    test_df.at[0,"Arrival_Time"] = Arrival_Time
    test_df.at[0,"Total_Stops"] = Total_Stops
    test_df.at[0,"Additional_Info"] = Additional_Info
    test_df.at[0,"Month_of_Journey"] = Month_of_Journey
    test_df.at[0,"Day_of_Journey"] = Day_of_Journey
    test_df.at[0,"Duration_min"] = Duration_min
    st.dataframe(test_df)
    result = Model.predict(test_df)[0]
    return result


def main():
    st.title("Flight Price Prediction")
    Airline = st.selectbox("Airline" , ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet',
       'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia', 'Trujet'])
    Source = st.selectbox("Source" , ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'])
    Destination = st.selectbox("Destination" , ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad'])
    Dep_Time = st.slider("Dep_Time" , min_value= 0 , max_value=23 , value=0,step=1)
    Arrival_Time = st.slider("Arrival_Time" , min_value= 0 , max_value=23 , value=0,step=1)
    Total_Stops = st.selectbox("Total_Stops" ,[0, 2, 1])
    Additional_Info = st.selectbox("Additional_Info" , ['No Info', 'In-flight meal not included', '1 Long layover',
       'Change airports', 'Red-eye flight'])
    Month_of_Journey = st.selectbox("Month_of_Journey" ,['March', 'May', 'June', 'April'])
    Day_of_Journey = st.selectbox("Day_of_Journey" ,['Sunday', 'Wednesday', 'Friday', 'Monday', 'Tuesday', 'Saturday',
       'Thursday'])
    Duration_min = st.slider("Duration_min" , min_value= 60 , max_value=1255 , value=0,step=1)
    
    
    if st.button("predict"):
        
        result = prediction(Airline, Source, Destination,Dep_Time, Arrival_Time,
       Total_Stops, Additional_Info, Month_of_Journey, Day_of_Journey,
       Duration_min)
        st.success(f'Fair Price will be around Rs. {result}')
        
if __name__ == '__main__':
        main()   
