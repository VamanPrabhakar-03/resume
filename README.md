AI-Powered Resume Analyzer

An AI-powered tool that extracts, analyzes, and ranks resumes based on their similarity to given job descriptions, using NLP and deep learning techniques. This application helps recruiters quickly shortlist candidates by providing AI-generated summaries and ranked recommendations.

🚀 Features
- Upload resumes in **PDF** or **DOCX** format.
- Automatic extraction of candidate details (Name, Email, Skills, Experience level, etc.).
- **BERT-based similarity scoring** between resumes and job descriptions.
- **AI-generated resume summaries** using Groq's LLaMA models.
- Candidate ranking based on similarity scores.
- **Admin dashboard** for viewing, sorting, and downloading candidate data.
- Data stored in **MySQL** for persistence.
- Deployed on **Google Cloud Platform (GCP)**.

 🛠 Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python
- **Database:** MySQL
- **NLP Models:** BERT (`sentence-transformers/all-MiniLM-L6-v2`)
- **AI Summarization:** Groq LLaMA API
- **Deployment:** Google Cloud Compute Engine

📂 Project Structure
├── Production.py # Main Streamlit app
├── utils/ # Helper functions
├── Logo/ # Project logos
├── Uploaded_Resumes/ # Uploaded resumes
├── requirements.txt # Python dependencies

## ⚙️ Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-resume-analyzer.git
   cd ai-resume-analyzer
   
Install dependencies:
pip install -r requirements.txt

Configure MySQL connection in Production.py.

Run the app:

streamlit run Production.py --server.port 8501

🔮 Future Enhancements
Multiple resume uploads.

Partial visibility of resumes for recruiters.

Auto-filtering and deletion of resumes not matching criteria.

Experience-based auto-selection.
