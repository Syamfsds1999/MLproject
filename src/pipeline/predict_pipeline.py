import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class predictpipeline:
    def __init__(self,features):
        model_path='artifacts/model.pkl'
        preprocessor_path='artifacts/preprpcessor.pkl'
        model=load_object(file_path=model_path)
        preprocessor=load_object(file_path=preprocessor_path)
        data_scaled=preprocessor.transform(features)
        preds=model.predict(data_scaled)
        return preds
    except Exception as e:
raise CustomException(e,sys)

