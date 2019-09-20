import pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.externals import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    
    #アプリ化したときにすぐに値段が出るように関数を定義
    def diamond_pred(param_ct,param_color, param_clarity, param_cut):
        #param_color = ''
        if param_color == 'D':
            param_color = 9
        elif param_color == 'E':
            param_color = 8
        elif param_color == 'F':
            param_color = 7
        elif param_color == 'G':
            param_color = 6
        elif param_color == 'H':
            param_color = 5
        elif param_color == 'I':
            param_color = 4
        elif param_color == 'J':
            param_color = 3
        elif param_color == 'K':
            param_color = 2
        elif param_color == 'L':
            param_color = 1
        
        #param_clarity = ''
        if param_clarity =='VVS1':
            param_clarity = 7
        elif param_clarity =='VVS2':
            param_clarity = 6
        elif param_clarity =='VS1':
            param_clarity = 5
        elif param_clarity =='VS2':
            param_clarity = 4
        elif param_clarity =='SI1':
            param_clarity = 3
        elif param_clarity =='SI2':
            param_clarity = 2
        elif param_clarity =='I1':
            param_clarity = 1
            
        #param_cut = ''
        if param_cut == 'EX':
            param_cut = 5
        elif param_cut == 'VG':
            param_cut = 4
        elif param_cut == 'G':
            param_cut = 3
        elif param_cut == 'FAIR':
            param_cut = 2
        elif param_cut == 'POOR':
            param_cut = 1
        
        pred = np.array([[param_ct,param_color, param_clarity, param_cut]])
        pred_poly = PolynomialFeatures(degree=3).fit(pred)
        pred_poly_2 = pred_poly.transform(pred)
        return pred_poly_2
        
    

        
    param_ct =request.form['ct']
    param_color =request.form['color']
    param_clarity =request.form['clarity']
    param_cut = request.form['cut']

    model_Ridge = joblib.load("./data/model_Ridge.pkl")
    DP = diamond_pred(param_ct,param_color, param_clarity, param_cut)
    pred_out = model_Ridge.predict(DP)
    result = '{0:.0f}'.format(pred_out[0]*10000)
    
    return render_template('result.html',result=result)


if __name__ == '__main__':
    app.run()

