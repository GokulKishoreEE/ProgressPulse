from flask import Flask ,redirect, url_for,render_template,request
import pickle
from huggingface_hub import InferenceClient
import os
import webbrowser
from dotenv import load_dotenv
with open('./trainml1.pkl','rb') as f:
    md = pickle.load(f)
app = Flask(__name__,template_folder='templates')
km=[]

@app.route('/')
def welcome():
    return render_template('index.html')
@app.route('/data')
def data():
    return render_template('analyse.html')
@app.route('/submit',methods=['POST','GET'])
def submit():
    cgpa = float(request.form['CGPA'])
    credits = int(request.form['Credits'])
    extraCurricular = int(request.form['ExtraCurricular'])
    projects = int(request.form['Projects'])
    selfStudy = int(request.form['SelfStudy'])
    assignment = request.form['Assignment']
    engagement = int(request.form['Engagement'])
    contribution = int(request.form['Contribution'])
    arr = [[cgpa,credits,extraCurricular,projects,selfStudy,engagement,assignment,contribution]]
    k=md.predict(arr)
    pr = k[0]
    assignp = assignment*100
    client = InferenceClient(provider="novita",api_key=os.getenv('HF_TOKEN1'))

    iparam = f'As a teacher you are speaking to a student. Generate a suggestion or feedback not more than 400 words for a student based on their CGPA {cgpa}, credits {credits}, extra-curriculars {extraCurricular}, projects {projects}, self-study hours {selfStudy}, interest in subjects {engagement} out of 10, assignment completion {assignp}, need of faculty contribution for the student is {contribution} out of 10, and improvement potential {pr} out of 10.'
    messages = [
	{ "role": "user", "content": iparam }
    ]
    stream = client.chat.completions.create(
    model=os.getenv('HF_MODEL'), 
	messages=messages, 
	max_tokens=500,
	stream=True
    )
    p=""
    for chunk in stream:
        p += chunk.choices[0].delta.content
    km.append(p)
    return render_template('resultdisplay.html',kx=p,spc=pr)

if __name__ =='__main__':
    app.run()
