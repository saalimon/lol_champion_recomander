from flask import Flask, render_template, request, jsonify

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'model')
from apriori import get
from prediction import pred
from knn_champion import get_knn
app = Flask(__name__)

messages = [{'title': 'Message One',
             'content': 'Message One Content'},
            {'title': 'Message Two',
             'content': 'Message Two Content'}
            ]

@app.route('/')
def index():
    return render_template('index.html', messages=messages)

@app.route('/apriori', methods = ['GET', 'POST'])
def apriori():
    if request.method == 'POST':
        query = request.form['query']
        rules_query, status = get(query)
        return render_template('apriori.html', tables=[rules_query[['antecedents', 'consequents']].head(30).to_html(classes='data', header="true")])
    return render_template('apriori.html')
@app.route('/predict',methods = ['GET','POST'])
def prediction():
    
    if request.method == 'POST':
        query = {
            "top_t1":request.form['top_t1'],
            "top_t2":request.form['top_t2'],
            "mid_t1":request.form['mid_t1'],
            "mid_t2":request.form['mid_t2'],
            "adc_t1":request.form['adc_t1'],
            "adc_t2":request.form['adc_t2'],
            "jg_t1":request.form['jg_t1'],
            "jg_t2":request.form['jg_t2'],
            "sup_t1":request.form['sup_t1'],
            "sup_t2":request.form['sup_t2']

        }
        out = pred(**query)        
        return render_template('prediction.html', winrate="{:.2f}".format(out[0][1]))
    return render_template('prediction.html')

@app.route('/knn',methods = ['GET','POST'])
def knn():
    # current_champ_list = ["JG_T1_Graves",'MID_T1_Ahri','ADC_T1_Caitlyn','TOP_T1_Volibear','SUP_T1_Lux']
    # opponent_champ_list = ['ADC_T2_Ezreal','SUP_T2_Leona','MID_T2_Azir','JG_T2_Shen','TOP_T2_Wukong']
    global current_c
    current_c = [""]*5
    global opponent_c
    opponent_c = [""]*5

    if request.method == 'POST':
        if request.form['submit_button'] == 'find champion':
            current_c = [request.form['t11'],request.form['t12'],request.form['t13'],request.form['t14'],request.form['t15']]
            opponent_c = [request.form['t21'],request.form['t22'],request.form['t23'],request.form['t24'],request.form['t25']]
            role = request.form['role']
            recommend_champions_df = get_knn(current_c, opponent_c, role)
            return render_template('knn.html', current = current_c, opponent = opponent_c, tables=[recommend_champions_df.to_html(classes='data', header="true")])
        else:
            
            current_c = [request.form['t11'],request.form['t12'],request.form['t13'],request.form['t14'],request.form['t15']]
            opponent_c = [request.form['t21'],request.form['t22'],request.form['t23'],request.form['t24'],request.form['t25']]
            if len(list(filter(None, current_c))) == 5 and len(list(filter(None, opponent_c))) == 5:
                query = {
                    "top_t1":current_c[0],
                    "top_t2":opponent_c[0],
                    "mid_t1":current_c[1],
                    "mid_t2":opponent_c[1],
                    "adc_t1":current_c[2],
                    "adc_t2":opponent_c[2],
                    "jg_t1":current_c[3],
                    "jg_t2":opponent_c[3],
                    "sup_t1":current_c[4],
                    "sup_t2":opponent_c[4]
                }
                out = pred(**query)        
                return render_template('prediction.html', winrate="{:.2f}".format(out[0][1]))
            else:
                return render_template('prediction.html')
    return render_template('knn.html', current = current_c, opponent = opponent_c)


@app.route('/send',methods = ['GET','POST'])
def send():
    if request.method == 'POST':
        age = request.form['age']
        return render_template('age.html',age=age)
    return render_template('index.html')
