'''logistic regression-api'''

from flask import Flask, redirect, url_for, request,jsonify
import joblib
app = Flask(__name__)

@app.route('/logisticregression',methods = ['POST'])
def logisticregression():
   if request.method == 'POST':
      Esalary= request.form['Esalary']
      Eage=request.form['Eage']
      cj= joblib.load('classifier_joblib.sav')
      Purchased =cj.predict([[Eage,Esalary]])
      print(Purchased[0])
      return jsonify(Purchased=str(Purchased[0]))

if __name__ == '__main__':
   app.run(debug = True)
