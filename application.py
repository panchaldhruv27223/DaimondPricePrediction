from flask import Flask, request, render_template, jsonify
from src.pipelines.prediction_pipeline import customData, predictionPipeline
import numpy as np

application = Flask(__name__)
app = application


@app.route("/")
def homePage():
    return render_template("index.html")



@app.route("/predict",methods = ["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else :
        data = customData(
            carat = np.float(request.form.get("carat")),
            depth = np.float(request.form.get("depth")),
            table = np.float(request.form.get("table")),
            x = np.float(request.form.get("x")),
            y = np.float(request.form.get("y")),
            z = np.float(request.form.get("z")),
            cut = request.form.get("cut"),
            color = request.form.get("color"),
            clarity = request.form.get("clarity")
        )
        print("data is : ",data)
        
        finalNewData = data.getDataAsDataFrame()
        prediction = predictionPipeline()

        predictOut = prediction.prediction(finalNewData)
        print(predictOut)
        
        # result = round(predictOut[0],2)

        # return render_template("form.html",final_result=result)
        return render_template("form.html",final_result=12)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug = True)
    
    
    
