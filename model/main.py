import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from pickle5 import pickle 

# model creation
def create_model(data):
    # target variable 
    y=data['diagnosis']
    # independent variable
    X=data.drop(['diagnosis'],axis=1)

    # scale the data
    scaler=StandardScaler()
    scaled_X=scaler.fit_transform(X)
    
    # split the data into train and test sets
    train_X,test_X,train_y,test_y = train_test_split(scaled_X,y,test_size=0.2,random_state=42)

    # train the model
    model = LogisticRegression()
    model.fit(train_X, train_y)

    # test the model
    y_predict=model.predict(test_X)

    print('accuracy of model :',accuracy_score(test_y,y_predict))

    print("classification report of model \n :",classification_report(test_y,y_predict))
    return model ,scaler


                                                
def get_data():
    data=pd.read_csv('/Users/vishalroy/Downloads/devloper/breast cancer detection/data.csv')
#    print(data.head())
    data=data.drop(['Unnamed: 32','id'],axis=1) # redundant column
    # convert target column 
    data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
    return data
    

def main():
    data=get_data()
    # check there any null values
    #print(data.info())
    print(data['diagnosis'].value_counts())
    model,scaler = create_model(data) # create model

    # dump the model and scaler for testing purposes
    with open('model.pkl','wb')as file:
        pickle.dump(model,file)

    with open('scaler.pkl','wb')as file:
        pickle.dump(scaler,file)

    
# #train the model
# train(model)

# #evaluate the model
# evaluate(model)

if __name__ == '__main__':
    main()



   
