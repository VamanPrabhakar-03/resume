import streamlit as st
import pandas as pd
import base64,random
import time,datetime
#libraries to parse the resume pdf files
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io,random
from streamlit_tags import st_tags
from PIL import Image
import pymysql
from Courses import ds_course,web_course,android_course,ios_course,uiux_course,resume_videos,interview_videos
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
os.environ["PAFY_BACKEND"] = "internal"
import pafy
import plotly.express as px #to create visualisations at the admin session
import nltk
import json
from pytube import YouTube
nltk.download('stopwords')


def fetch_yt_video(link):
    try:
        yt = YouTube(link)
        return yt.title
    except Exception as e:
        print(f"Error fetching video: {e}")
        return "Video title unavailable"


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
def calculate_similarity_tfidf(resume_text, job_description):
    """
    Takes resume text and job description, and returns cosine similarity score.
    """
    documents = [resume_text, job_description]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(float(cosine_sim[0][0]), 2)  # return similarity score rounded to 2 decimals

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 🎓**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course

#CONNECT TO DATABASE

connection = pymysql.connect(host='localhost',user='root',password='Grizz*0304*',db='cv')
cursor = connection.cursor()

def insert_data(name, email, timestamp, no_of_pages, cand_level, skills, similarity_score):
     insert_sql = ("INSERT INTO user_data (Name, Email_ID, Timestamp, Page_no, User_level, Actual_skills, Similarity_Score) "
                   "VALUES (%s,%s,%s,%s,%s,%s,%s)")
     rec_values = (name, email, timestamp, no_of_pages, cand_level, skills, similarity_score)

     cursor.execute(insert_sql, rec_values)
     connection.commit()

st.set_page_config(
    page_title="AI-Powered Resume Analyzer",
    page_icon='./Logo/logo2.png',
)
def run():
    img = Image.open('./Logo/ResumeAi.png')
    # img = img.resize((250,250))
    st.image(img)
    st.title("AI Resume Analyser")
    st.sidebar.markdown("# Choose User")
    activities = ["User","Admin"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    link = '[©Developed by Mr Vaman](https://www.linkedin.com/in/vaman-prabakar-32b6072a1/)'
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
         User_level BLOB NOT NULL,
         Actual_skills BLOB NOT NULL,
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
                st.success("Hello "+ resume_data['name'])
                st.subheader("**Your Basic info**")
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


                keywords = st_tags(label='### Your Current Skills',
                                   value=resume_data['skills'],key = '1  ')



                ## Insert into table
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date+'_'+cur_time)


                resume_score = 0

                st.subheader("**Resume Score📝 (TF-IDF based)**")

                ## TEMPORARILY HARDCODE A JOB DESCRIPTION FOR TESTING
                job_description_text = """
                                We are looking for a Data Scientist with experience in Machine Learning, Python, Deep Learning,
                                and Data Visualization. Proficiency with TensorFlow, PyTorch and Pandas required.
                                """
                similarity_score = calculate_similarity_tfidf(resume_text, job_description_text)
                similarity_score = round(similarity_score, 2)

                st.success(f"🧠 Similarity Score: {similarity_score * 100}%")
                st.warning("** Note: This score is calculated based on the content that you have in your Resume. **")

                insert_data(
                    resume_data['name'],
                    resume_data['email'],
                    timestamp,
                    str(resume_data['no_of_pages']),
                    json.dumps(cand_level),  # This is fine if it's a simple string
                    str(similarity_score),
                    json.dumps(resume_data['skills'], ensure_ascii=False)  # 🔥 human-readable
                )




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
                                                'User Level', 'Actual Skills','Similarity_Score'])
                st.dataframe(df)
                st.markdown(get_table_download_link(df,'User_Data.csv','Download Report'), unsafe_allow_html=True)
                ## Admin Side Data
                query = 'select * from user_data;'
                plot_data = pd.read_sql(query, connection)
                plot_data['Actual_skills'] = plot_data['Actual_skills'].apply(
                    lambda x: json.loads(x.decode('utf-8')) if isinstance(x, bytes) else json.loads(x)
                )
    
                ## Pie chart for predicted field recommendations




            else:
                st.error("Wrong ID & Password Provided")


run()