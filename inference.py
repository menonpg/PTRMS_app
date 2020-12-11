
import config
# import cv2
import h2o
import pandas as pd
h2o.init(port = 54321, ip = "localhost", bind_to_localhost = False, max_mem_size='1G')

h2o.remove_all()

def inference(model, myCSV, threshold = 0.920060452296198):
    model_name = f"{config.MODEL_PATH}{model}"
    model = h2o.load_model(model_name)
    
    h2odf = h2o.H2OFrame(pd.read_csv(myCSV),  destination_frame="testData.hex")
    df = h2odf.as_data_frame()
    predictions = model.predict(h2odf )
    # df['alert_h2o'] = predictions.as_data_frame().predict
    df['Probability_COVID19'] = predictions.as_data_frame().iloc[:,2]
    df['COVID19_Status']=df['Probability_COVID19'].map(lambda x: 1 if x <= threshold else 0)
    df['Probability_COVID19'] = 1-df['Probability_COVID19']
    cols = df.columns.tolist()
    df = df[cols[-2:] + cols[:-2]]
    
    return df 