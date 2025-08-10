import base64
import datetime
import io
import os
import time
import pandas as pd
import pymysql
import streamlit as st
from PIL import Image
from pdfminer3.converter import TextConverter
from pdfminer3.layout import LAParams
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfpage import PDFPage
# libraries to parse the resume pdf files
from pyresparser import ResumeParser
from sentence_transformers import SentenceTransformer, util
os.environ["PAFY_BACKEND"] = "internal"
import nltk
import json
nltk.download('stopwords')
import openai

client = openai.OpenAI(
    api_key="GROQ API Key",
    base_url="https://api.groq.com/openai/v1"
)
# Sample Job Descriptions
job_descriptions = [
    "We are looking for a Data Scientist with experience in Machine Learning, Python, Deep Learning, and Data Visualization. Proficiency with TensorFlow, PyTorch, and Pandas required. Strong statistical knowledge and problem-solving skills are essential.",

    "Seeking a skilled Frontend Developer proficient in React.js, JavaScript, HTML, and CSS. Experience with REST APIs, responsive design, and version control (Git) is a must. Knowledge of UI/UX best practices is desirable.",

    "Looking for a Backend Developer with strong expertise in Python, Django/Flask, and database management using MySQL or PostgreSQL. Familiarity with APIs, cloud platforms, and microservices is a plus.",

    "Hiring an Android Developer with hands-on experience in Java, Kotlin, Android Studio, and material design principles. Understanding of REST APIs and push notifications is essential.",

    "We are seeking an iOS Developer proficient in Swift, Xcode, and UIKit. Strong understanding of iOS architecture patterns, RESTful services, and Apple's design guidelines is required.",

    "We are looking for a creative UI/UX Designer with proficiency in Figma, Adobe XD, Sketch, and user research methods. Experience in wireframing, prototyping, and usability testing is necessary.",

    "Hiring a Full Stack Developer skilled in React.js, Node.js, MongoDB, and Express.js. Knowledge of DevOps, CI/CD pipelines, and cloud services (AWS/GCP) will be advantageous.",

    "We are looking for an AI/ML Engineer with experience in building machine learning models using Scikit-learn, TensorFlow, or PyTorch. Strong Python programming, statistics, and data preprocessing skills are mandatory."
]


def get_table_download_link(df,filename,text):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()
    return text

# Load BERT model (only once)
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def calculate_similarity_bert(resume_text, job_description):
    embeddings = model.encode([resume_text, job_description], convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return round(float(cosine_sim), 2)


def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)



#CONNECT TO DATABASE

connection = pymysql.connect(host='localhost',user='root',password='password',db='cv')
cursor = connection.cursor()


def insert_data(name, email, timestamp, no_of_pages, cand_level, similarity_score, skills, rank):
    insert_sql = """INSERT INTO user_data 
    (Name, Email_ID, Timestamp, Page_no, User_level, Similarity_Score, Actual_skills,`Rank`)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""

    rec_values = (name, email, timestamp, no_of_pages, cand_level, similarity_score, skills, rank)
    cursor.execute(insert_sql, rec_values)
    connection.commit()



st.set_page_config(
    page_title="AI-Powered Resume Analyzer",
    page_icon='./Logo/Logo2.png',
)

# Load summarizer once
def generate_summary_with_groq(resume_text):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system",
             "content": "You are an expert resume summarizer. Summarize the resume content in a concise, professional way."},
            {"role": "user", "content": f"Summarize this resume:\n\n{resume_text}"}
        ],
        max_tokens=300,
        temperature=0.5
    )
    summary = response.choices[0].message.content.strip()
    return summary




def run():
    img = Image.open('./Logo/ResumeAi.png')
    #img = img.resize((250,250))
    st.image(img)
    st.title("AI Resume Analyser")
    st.sidebar.markdown("# Choose User")
    activities = ["User","Admin"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    link = '[©Developed by Vaman Prabhakar](https://www.linkedin.com/in/vaman-prabakar-32b6072a1/)'
    st.sidebar.markdown(link, unsafe_allow_html=True)

    # Create the DB
    db_sql = """CREATE DATABASE IF NOT EXISTS CV;"""
    cursor.execute(db_sql)

    # Create table
    DB_table_name = 'user_data'
    table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
        (ID INT NOT NULL AUTO_INCREMENT,
    Name VARCHAR(500) NOT NULL,
    Email_ID VARCHAR(500) NOT NULL,
    Timestamp VARCHAR(50) NOT NULL,
    Page_no VARCHAR(5) NOT NULL,
    User_level VARCHAR(500) NOT NULL,        -- FIXED: VARCHAR instead of BLOB
    Actual_skills TEXT NOT NULL,             -- TEXT or VARCHAR depending on expected length
    Similarity_Score VARCHAR(10) NOT NULL,
    PRIMARY KEY (ID));
    """
    cursor.execute(table_sql)

    if choice == 'User':
        st.markdown('''<h5 style='text-align: left; color: #021659;'> Upload your resume, and get smart recommendations</h5>''',
                    unsafe_allow_html=True)
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf", "docx"])
        if pdf_file is not None:
            with st.spinner('Uploading your Resume...'):
                time.sleep(4)
            save_image_path = './Uploaded_Resumes/'+pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)
            resume_data = ResumeParser(save_image_path).get_extracted_data()
            if resume_data:
                ## Get the whole resume data
                resume_text = pdf_reader(save_image_path)

                st.header("**Resume Analysis**")
                st.subheader("**Basic info**")
                try:
                    st.text('Name: '+resume_data['name'])
                    st.text('Email: ' + resume_data['email'])
                    st.text('Contact: ' + resume_data['mobile_number'])
                    st.text('Resume pages: '+str(resume_data['no_of_pages']))
                except:
                    pass
                cand_level = ''
                if resume_data['no_of_pages'] == 1:
                    cand_level = "Fresher"
                    st.markdown( '''<h4 style='text-align: left; color: #d73b5c;'>You are at Fresher level!</h4>''',unsafe_allow_html=True)
                elif resume_data['no_of_pages'] == 2:
                    cand_level = "Intermediate"
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',unsafe_allow_html=True)
                elif resume_data['no_of_pages'] >=3:
                    cand_level = "Experienced"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',unsafe_allow_html=True)



                ## Insert into table
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date+'_'+cur_time)
                st.subheader("Choose Job Description to Match:")

                selected_jd = st.selectbox("Select a Job Role:", job_descriptions)

                if selected_jd:
                    similarity_score = calculate_similarity_bert(resume_text, selected_jd)
                    similarity_score = round(similarity_score, 2)

                    st.success(f"🧠 Similarity Score: {similarity_score * 100}%")
                    # Example: Simple rank based on similarity score
                    if similarity_score >= 0.8:
                        rank = 1  # Highly matching
                    elif similarity_score >= 0.5:
                        rank = 2  # Medium match
                    else:
                        rank = 3  # Low match

                    resume_summary = generate_summary_with_groq(resume_text)
                    st.subheader("**Resume Summary:**")
                    st.success(resume_summary)

            # Display Save Button
            if st.button("Save My Resume Analysis"):
                insert_data(
                    resume_data['name'],
                    resume_data['email'],
                    timestamp,
                    str(resume_data['no_of_pages']),
                    json.dumps(cand_level),
                    str(similarity_score),
                    json.dumps(resume_data['skills']),
                    rank
                )
                st.success("✅ Your data has been successfully saved!")

    else:
        ## Admin Side
        st.success('Welcome to Admin Side')
        # st.sidebar.subheader('**ID / Password Required!**')

        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')
        if st.button('Login'):
            if ad_user == 'vaman' and ad_password == 'vaman123':
                st.success("Welcome Vaman")
                # Display Data
                cursor.execute('''SELECT*FROM user_data''')
                data = cursor.fetchall()
                st.header("**User's Data**")
                df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Timestamp', 'Total Page',
                                                 'User Level', 'Actual Skills', 'Similarity_Score', 'Rank'])
                st.dataframe(df)

                st.markdown(get_table_download_link(df,'User_Data.csv','Download Report'), unsafe_allow_html=True)
                ## Admin Side Data
                query = 'select * from user_data;'
                plot_data = pd.read_sql(query, connection)
                plot_data['Actual_skills'] = plot_data['Actual_skills'].apply(
                    lambda x: json.loads(x.decode('utf-8')) if isinstance(x, bytes) else json.loads(x)
                )





            else:
                st.error("Wrong ID & Password Provided")


run()

