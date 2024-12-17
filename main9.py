import streamlit as st
from streamlit_ace import st_ace
import tempfile
import ollama
import pandas as pd
from pyopenproject.openproject import OpenProject
from pyopenproject.model.work_package import WorkPackage
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import subprocess
from utils2 import (
    run_python_code, run_sql_code, format_python_code, format_sql_code,
     run_cpp_code, run_java_code,
    format_cpp_code, format_java_code,store_code,
    store_feedback,store_ticket,fetch_jira_data
)

# from code_assistant_app import fix_bug


##########
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate


##############

st.set_page_config(layout="wide",page_icon='🐦')

if 'task' not in st.session_state:
    st.session_state['task'] = 'No task Specified'

if 'task_title' not in st.session_state:
    st.session_state['task_title'] = 'No task Specified'

if 'task_description' not in st.session_state:
    st.session_state['task_description'] = 'No task Specified'

if 'task_ac' not in st.session_state:
    st.session_state['task_ac'] = 'No task Specified'

def generate_task_title(code):
    response = ollama.chat(model='llama3.2:1b', stream=True, messages=[
        {"role": "user", "content": f"Provide one best title without any explanation or description for the task :\n{code}"}
    ])
    description = ""
    for partial_resp in response:
        token = partial_resp["message"]["content"]
        description += token
    return description




def generate_task_description(code):
    response = ollama.chat(model='llama3.2:1b', stream=True, messages=[
        {"role": "user", "content": f"Provide a description of the following task in short:\n{code}"}
    ])
    description = ""
    for partial_resp in response:
        token = partial_resp["message"]["content"]
        description += token
    return description

def generate_task_ac(code):
    response = ollama.chat(model='llama3.2:1b', stream=True, messages=[
        {"role": "user", "content": f"Provide an acceptance criteria with sample input and output without any code for the following task :\n{code}"}
    ])
    description = ""
    for partial_resp in response:
        token = partial_resp["message"]["content"]
        description += token
    return description

def display_jira_data(status):
    try:
        df = fetch_jira_data(status)
        if df.empty:
            st.warning("No Tickets are present in this Section.")
        else:
            if 'TITLE' not in df.columns:
                st.error("The table must have a 'TITLE' column to use as expander headers.")
                return
            for _, row in df.iterrows():
                title = row['TITLE']
                with st.expander(f"{title}"):
                    for column, value in row.items():
                        st.write(f"**{column}:** {value}")

    except Exception as e:
        st.error(f"Error fetching data: {e}")

def run_ollama(prompt):
    """
    Use the Ollama CLI to generate a response from the model.
    """
    try:
        process = subprocess.Popen(
            ["ollama", "run", "codegemma:7b"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Ensures UTF-8 encoding
        )
        
        stdout, stderr = process.communicate(input=prompt)

        if stderr:
            print(f"Error from Ollama CLI: {stderr.strip()}")
            return None

        return stdout.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None      

def display_output(result):
    if isinstance(result, pd.DataFrame):
        if result.empty:
            st.session_state["output"] = "The result DataFrame is empty."
        else:
            # Display as DataFrame if it's too large, provide an option to view more
            st.session_state["output"] = result
            #st.write(result)
            # Optionally display a message if it's large
            #if result.shape[0] > 20 or result.shape[1] > 10:  # Adjust size as per need
                #st.write("Note: This is a large table. You can scroll for more data.")
    elif isinstance(result, str):
        st.session_state["output"] = result
    elif isinstance(result, (int, float)):
        st.session_state["output"] = f"Output: {result}"
    else:
        # For unsupported types, just show the type
        st.session_state["output"] = f"Unsupported output type: {type(result)}"

# Initialize OpenProject with URL and API key
op = OpenProject(
    url="https://phoenix04.openproject.com/",
    api_key="4410371acbbe022b9465857cb0032e3d718c42bb82bb01ccdb9cd226b29ef7d1"
)

# Get the project service
ps = op.get_project_service()

# Retrieve all projects
projects = ps.find_all()

# Find the desired project by name
proj = next((p for p in projects if p.name == "Phoenix_hack"), None)
if proj is None:
    st.error("Project 'Phoenix_hack' not found.")
    st.stop()
 
# Fetch all priorities
priorities_service = op.get_priority_service()
priorities = priorities_service.find_all()

# Fetch all users (assignees)
users_service = op.get_user_service()
users = users_service.find_all()

# Define valid statuses with mapping
status_map = {
    "New": 1,
    "In progress": 7,
    "On hold": 13,
    "Closed": 12,
    "Rejected": 14
}

# Display available priorities and users
priority_options = {priority.id: priority.name for priority in priorities}
user_options = {user.id: user.name for user in users}

class DummyWorkPackage:
    def __init__(self, work_package_id):
        self.id = work_package_id

import streamlit as st
from streamlit_lottie import st_lottie
import requests

# Function to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations/icons
phoenix_animation = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_w51pcehl.json")
# import streamlit as st

# Custom CSS for styling
st.markdown(
    """
    <style>
        .center {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .custom-button {
            border: none;
            color: white;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
        .generate-button {
            background-color: #FF5722; /* Phoenix theme orange */
        }
        .view-button {
            background-color: #03A9F4; /* Sky blue */
        }
        .editor-button {
            background-color: #4CAF50; /* Green */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main Page Code
def main_page():
    # Page Header
    # st.markdown("<h1 style='text-align: center; color: #FF5722;'>🔥 Phoenix Platform 🔥</h1>", unsafe_allow_html=True)
    # st_lottie(phoenix_animation, height=100, key="phoenix")
    
    col1, col2 = st.columns([2, 5])  # Adjust the column width ratios as needed

    with col1:
        st_lottie(phoenix_animation, height=200, key="phoenix")

    with col2:
        st.markdown(
            "<h1 style='text-align: left; color: #FF5722;'>🔥 Phoenix Platform 🔥</h1>", 
            unsafe_allow_html=True
            
        )
        st.markdown("<h3 style='text-align: left; color: #757575;'>Welcome to your productivity hub!</h3>", unsafe_allow_html=True)
 
    # st.markdown("<h3 style='text-align: center; color: #757575;'>Welcome to your productivity hub!</h3>", unsafe_allow_html=True)
    # st.markdown("---")

    # Task Input Section
    # st.markdown("### Your Goals for the Session:")
    # st.markdown("**What are your plans for this session?**")
    st.session_state["task"] = st.text_area(
        "What are your plans for this session?",
        placeholder="Describe your tasks or goals for today...",
    )
    # st.markdown("---")

    # Buttons Section
    col11, col12, col3 = st.columns(3)
    with col11:
        if st.button("Generate Ticket", key="generate", help="Create a new ticket based on your task"):
            if st.session_state["task"]:
                # Generate Ticket Logic
                st.session_state['task_title'] = generate_task_title(st.session_state["task"])
                st.session_state['task_description'] = generate_task_description(st.session_state["task"])
                st.session_state['task_ac'] = generate_task_ac(st.session_state["task"])
                st.success("Ticket Generated Successfully!")
                default_status = status_map['New']  # Assuming 1 corresponds to 'New'
                default_priority = next(priority for priority in priorities if priority.name.lower() == "low")
                combined_description = (
                    f"**Task Description:**\n{st.session_state['task_description']}\n\n"
                    f"**Acceptance Criteria:**\n{st.session_state['task_ac']}"
                 )
                wp_data = {
                    "subject": st.session_state['task_title'],
                    "description": {
                        "raw":combined_description,
                         
                    },
                    "_links": {
                        "status": {"href": f"/api/v3/statuses/{default_status}"},
                        "priority": {"href": f"/api/v3/priorities/{default_priority.id}"},
                        #"assignee": {"href": f"/api/v3/users/{default_assignee.id}"}
                    }
                }
                wp_data_obj = WorkPackage(json_obj=wp_data)
                work_package = ps.create_work_package(proj, wp_data_obj)
                
                # Display Ticket Details
                with st.expander("View Ticket Details"):
                    st.markdown(f"**Title:** {st.session_state['task_title']}")
                    st.markdown(f"**Description:** {st.session_state['task_description']}")
                    st.markdown(f"**Acceptance Criteria:** {st.session_state['task_ac']}")

                # Store Ticket Logic
                store_ticket(
                    st.session_state['task_title'], 
                    st.session_state['task_description'], 
                    st.session_state['task_ac'], 
                    'UNASSIGNED', 
                    'TO DO', 
                    'MEDIUM', 
                    ''
                )
            else:
                st.error("Please enter a task description to generate a ticket.")

    with col12:
        if st.button("View Tickets", key="view", help="Check all created tickets"):
            st.session_state["current_page"] = "Jira2"
            st.rerun()

    with col3:
        if st.button("Go to Editor", key="editor", help="Navigate to the code editor"):
            st.session_state["current_page"] = "first"
            st.rerun()

  # Replace with a phoenix animation link

 

def first_page():
    if st.sidebar.button("Back"):
        st.session_state["current_page"] = "home"  # Change page to home
        st.rerun()  # Trigger rerun to reload the app and navigate to the home page

    st.markdown("<h3>🐦 PHOENIX - Code Editor</h3>", unsafe_allow_html=True)

    languages = ["Python", "SQL", "C++", "Java"]

    if "code" not in st.session_state:
        st.session_state["code"] = ""
    if "custom_input" not in st.session_state:
        st.session_state["custom_input"] = ""
    if "output" not in st.session_state:
        st.session_state["output"] = ""
    if "selected_language" not in st.session_state:
        st.session_state["selected_language"] = "Python"
    if "code_description_response" not in st.session_state:
        st.session_state["code_description_response"] = ""
    if "conversion_language" not in st.session_state:
        st.session_state["conversion_language"] = ""
    if "action_option" not in st.session_state:
        st.session_state["action_option"] = ""    
    if "completed_code" not in st.session_state:
        st.session_state["completed_code"] = ""

    if "optimized_code" not in st.session_state:
        st.session_state["optimized_code"] = ""

    if "reviewed_code" not in st.session_state:
        st.session_state["reviewed_code"] = ""   
        
    def call_run(name):
        result = None
        if name == 'run_button':
            if st.session_state["selected_language"] == "SQL":
                result = run_sql_code(st.session_state["code"])
            elif st.session_state["selected_language"] == "Python":
                result = run_python_code(st.session_state["code"], st.session_state["custom_input"])
            elif st.session_state["selected_language"] == "C++":
                result = run_cpp_code(st.session_state["code"], st.session_state["custom_input"])
            elif st.session_state["selected_language"] == "Java":
                result = run_java_code(st.session_state["code"], st.session_state["custom_input"])
        return result


    def run_func():
        r = call_run('run_button')
        # st.session_state["output"] = str(r) if r else "No output produced or an error occurred."
        display_output(r)
    ############
    model = OllamaLLM(model="codellama")

    completion_template = """
    You are an AI for Python code completion. Your task is to complete the given Python code snippet. Ensure the code is syntactically and logically correct.

    Code Snippet: {context}

    Complete the code:
    """

    optimization_template = """
    Analyze and improve the following Python code to make it more efficient, concise, and Pythonic while adhering to best practices. Focus on eliminating redundant operations, improving readability, and maintaining functionality. Provide the optimized code only, without explanations or comments.

    Code Snippet: {context}

    Optimize the code:
    """

    review_template = """
    You are an AI for reviewing Python code. Your task is to find potential bugs, logical errors, or stylistic issues in the given Python code snippet, and suggest fixes or improvements.

    Code Snippet: {context}

    Review and fix the code:
    """

    completion_prompt = ChatPromptTemplate.from_template(template=completion_template)
    optimization_prompt = ChatPromptTemplate.from_template(template=optimization_template)
    review_prompt = ChatPromptTemplate.from_template(template=review_template)

    completion_chain = completion_prompt | model
    optimization_chain = optimization_prompt | model
    review_chain = review_prompt | model

    def complete_code(code_snippet):
        try:
            result = completion_chain.invoke({"context": code_snippet})
            if isinstance(result, str):
                return result.strip()
            else:
                return "Unexpected result format."
        except Exception as e:
            return f"Error generating code: {e}"

    def optimize_code(code_snippet):
        try:
            result = optimization_chain.invoke({"context": code_snippet})
            if isinstance(result, str):
                return result.strip()
            else:
                return "Unexpected result format."
        except Exception as e:
            return f"Error optimizing code: {e}"

    def review_code(code_snippet):
        try:
            result = review_chain.invoke({"context": code_snippet})
            if isinstance(result, str):
                return result.strip()
            else:
                return "Unexpected result format."
        except Exception as e:
            return f"Error reviewing code: {e}"    
    ##############   
    
    editor_col, buttons_col = st.columns([3, 1])

    with editor_col:
        # col1, col2, col3, col4,col5 = st.columns(5)
        col1, col2,col6,col7,col8,col9, col3, col4,col5 = st.columns([1,1,1,1,1,1,1,1,2])
        with col2:
            if st.button("▶️", key='run_button',help="Run"):
                run_func()
        with col1:
            if st.button("🆘",help="Help!"): #Help!
                st.session_state["current_page"] = "chat"  # Change page to chat page
                st.rerun()  # Trigger rerun to reload the app and navigate to chat page
        with col6:
            suggest_code = st.button("💡",help="Suggest code")
            if suggest_code:
                    st.session_state["action_option"]="Code Suggest"
        with col7:
            optimize_button = st.button("⌛",help="Optimized")
            if optimize_button:
                code_to_optimize = {st.session_state['code']}
                if code_to_optimize:
                    optimized_code= optimize_code(code_to_optimize)
                    st.session_state["optimized_code"]=optimized_code
                    st.session_state["action_option"]="Optimize Code"
                else:
                    st.warning("Please enter some code to optimize!")
        with col8:
            fix_code = st.button("🛠️",help="Fix Code")
            if fix_code:
                code_to_review = {st.session_state['code']}
                if code_to_review:
                    reviewed_code = review_code(code_to_review)
                    st.session_state["reviewed_code"]=reviewed_code
                    st.session_state["action_option"]="Fix Code"
                        # st.success("Code review and fixes completed!")
                else:
                    st.warning("Please enter some code to review and fix!")

        with col9:
            code_compt=st.button("🧩",help="Code Completion")
            if code_compt:
                code_to_complete = {st.session_state['code']}
                if code_to_complete:
                    completed_code = complete_code(code_to_complete)
                    st.session_state["completed_code"]=completed_code
                    st.session_state["action_option"]="Complete Code"  
                else:
                    st.warning("Please enter some code to complete!")

        with col3:
            ####
            def get_file_extension(language):
                if language.lower() == 'python':
                    return ".py"
                elif language.lower() == 'c++':
                    return ".cpp"
                elif language.lower() == 'sql':
                    return ".sql" 
                elif language.lower() == 'java':
                    return ".java"
                else:
                    return ".txt"
            
            if st.download_button(
                "⬇️", help="Download", 
                data=st.session_state['code'], 
                file_name=f"code{get_file_extension(st.session_state['selected_language'])}", 
                mime="text/plain"
            ):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(st.session_state['selected_language']))
                with open(temp_file.name, "w") as f:
                    f.write(st.session_state['code'])
            
            
        with col4:
            if st.button("🚩",help="Submit"): #🏁
                #code summit
                store_code( st.session_state['task_title'],st.session_state['task_description'],st.session_state['task_ac'],'UNASSIGNED','TO DO','MEDIUM',st.session_state["code"])

                st.toast("Success!", icon="✅")

        with col5:
            st.session_state["custom_input"]=st.text_input("Custom Input", placeholder="Enter your test cases here...")

        st.session_state["code"] = st_ace(
            value=st.session_state["code"],
            language=st.session_state["selected_language"].lower() if st.session_state["selected_language"] in ["Python", "SQL"] else "c_cpp" if st.session_state["selected_language"] == "C++" else "java",
            # theme="github", 
            theme="monokai",
            height=250,
            placeholder="Write your code here...",
            key="editor"
        )

        with buttons_col:
            st.session_state["selected_language"] = st.selectbox("Language", languages, index=languages.index(st.session_state["selected_language"]))
            # st.session_state["custom_input"]=st.text_input("Custom Input", placeholder="Enter your test cases here...")
            # st.text_area("Output", st.session_state["output"], placeholder="Run the code to see output...")
            with st.expander("Code Description"):
                def generate_description(code):
                    response = ollama.chat(model='llama3.2:1b', stream=True, messages=[
                        {"role": "user", "content": f"Provide a description of the following code in short:\n{code}"}
                    ])
                    description = ""
                    for partial_resp in response:
                        token = partial_resp["message"]["content"]
                        description += token
                    return description
                # Generate code description and update in session state
                description = generate_description(st.session_state["code"])
                st.session_state["code_description_response"] = description
                st.write(st.session_state["code_description_response"])
            
            st.session_state["converted_code"] = '' 
            with st.expander("Language Converter"):
                if st.session_state["code"]:
                    col1,col2,col4,col3 = st.columns([1,1,1,2])
                    with col1:
                        if st.button("🐍",help="Python"):
                            prompt = f"Convert the following {st.session_state['selected_language']} code to Python:\n{st.session_state['code']}"
                            converted_code = run_ollama(prompt)
                            st.session_state["converted_code"] = converted_code
                            st.session_state["action_option"]='Code Con'
                    with col2:
                        if st.button("☕",help="java"):
                            prompt = f"Convert the following {st.session_state['selected_language']} code to java which can executable:\n{st.session_state['code']}"
                            converted_code = run_ollama(prompt)
                            st.session_state["converted_code"] = converted_code
                            st.session_state["action_option"]='Code Con'
                    # col3,col4 = st.columns(2)
                    with col3:
                        if st.button("＋＋",help="c++"):
                            prompt = f"Convert the following {st.session_state['selected_language']} code to c++ code which can executable:\n{st.session_state['code']}"
                            converted_code = run_ollama(prompt)
                            st.session_state["converted_code"] = converted_code
                            st.session_state["action_option"]='Code Con'
                    with col4:
                        if st.button("💾",help="sql"):
                            prompt = f"Convert the following {st.session_state['selected_language']} code to sql:\n{st.session_state['code']}"
                            converted_code = run_ollama(prompt)
                            st.session_state["converted_code"] = converted_code
                            st.session_state["action_option"]='Code Con'
                else:
                    st.warning("Please enter some code to convert.")    
            
            # col1,col2 = st.columns(2)
            # with col1:
            #     suggest_code = st.button("Suggest code")
            #     if suggest_code:
            #         st.session_state["action_option"]="Code Suggest"
            # with col2:
            #     test_case = st.button("Test Cases")
            #     if test_case:
            #         st.toast("Success!",icon="✅")
                    
            # col3,col4 = st.columns(2)
            # with col3:
            #     optimize_button = st.button("Optimized")
            #     if optimize_button:
            #         # st.session_state["action_option"]="Optimize Code"
            #         code_to_optimize = {st.session_state['code']}
            #         if code_to_optimize:
            #             optimized_code= optimize_code(code_to_optimize)
            #             st.session_state["optimized_code"]=optimized_code
            #             st.session_state["action_option"]="Optimize Code"
            #         else:
            #             st.warning("Please enter some code to optimize!")
                
                
                    
            # with col4:
            #     fix_code = st.button("Fix Code")
            #     if fix_code:
            #         # st.toast("Success!", icon="✅")
            #         code_to_review = {st.session_state['code']}
            #         if code_to_review:
            #             reviewed_code = review_code(code_to_review)
            #             st.session_state["reviewed_code"]=reviewed_code
            #             st.session_state["action_option"]="Fix Code"
            #                 # st.success("Code review and fixes completed!")
            #         else:
            #             st.warning("Please enter some code to review and fix!")
                    
            # code_compt=st.button("Code Completion")
            # if code_compt:
            #     code_to_complete = {st.session_state['code']}
            #     if code_to_complete:
            #         completed_code = complete_code(code_to_complete)
            #         st.session_state["completed_code"]=completed_code
            #         st.session_state["action_option"]="Complete Code"  
            #     else:
            #         st.warning("Please enter some code to complete!")
            #     # st.toast("Success!", icon="✅")    
                
        
    # st.session_state["custom_input"]=st.text_input("Custom Input", placeholder="Enter your test cases here...")
    col21,col22=st.columns(2)
    with col21:
        st.write("Output:")
        if "output" in st.session_state:
            st.write(st.session_state["output"])
    with col22:
        if  st.session_state["action_option"]=="Code Con":
            # with st.expander("Converted Code", expanded=True):
            st.code(st.session_state["converted_code"], language=("Python").lower())   
        # st.code(st.session_state["optimized_code"], language='python')
        # code_space=st.text_area("Enter Your Code Snippet:", st.session_state["optimized_code"])
        # code_space=st.code(st.session_state["optimized_code"], language='python')
        elif st.session_state["action_option"] == "Optimize Code":
            if st.session_state["optimized_code"].startswith("Error"):
                st.error(st.session_state["optimized_code"]) 
            else:
                st.code(st.session_state["optimized_code"], language='python')
                
        elif st.session_state["action_option"] == "Fix Code":
            if st.session_state["reviewed_code"].startswith("Error"):
                st.error(st.session_state["reviewed_code"])
            else:
                st.code(st.session_state["reviewed_code"],language='python')
            # st.write("**Completion Notes:** Code completion is based on the input provided and common coding patterns.")
        elif st.session_state["action_option"]=="Complete Code" :
            if st.session_state["completed_code"].startswith("Error"):
                st.error(st.session_state["completed_code"])
            else:
                st.code(st.session_state["completed_code"], language='python')
                # st.success("Code completion generated!")
        elif st.session_state["action_option"]=="Code Suggest":
            st.subheader("Code Suggest")
            def generate_sugg(code):
                response = ollama.chat(model='llama3.2:1b', stream=True, messages=[
                    {"role": "user", "content": f"Provide a sample code for this:\n{code}"}
                ])
                description = ""
                for partial_resp in response:
                    token = partial_resp["message"]["content"]
                    description += token
                return description
            # Generate code description and update in session state
            sugg = generate_sugg(st.session_state["task"])
            st.code(sugg, language='python')
            # st.session_state["code_description_response"] = sugg
            # st.write(st.session_state["code_description_response"])
                
          


                     
    
    uploaded_file = st.sidebar.file_uploader("Browse file", type=["py", "sql", "cpp", "java"], label_visibility='collapsed')
    if uploaded_file is not None:
        st.session_state["code"] = uploaded_file.read().decode("utf-8")
       
    with st.sidebar.expander("Ticket"):
        st.write(st.session_state['task_title'])
        st.markdown("<h3>Description</h3>", unsafe_allow_html=True)
        st.write(st.session_state['task_description'])
        st.markdown("<h3>Acceptance criteria</h3>", unsafe_allow_html=True)
        st.write(st.session_state['task_ac']) 
    

    def format_code():
        if st.session_state["selected_language"] == "SQL":
            st.session_state["code"] = format_sql_code(st.session_state['code'])
        elif st.session_state["selected_language"] == "Python":
            st.session_state["code"] = format_python_code(st.session_state['code'])
        elif st.session_state["selected_language"] == "C++":
            st.session_state["code"] = format_cpp_code(st.session_state['code'])
        elif st.session_state["selected_language"] == "Java":
            st.session_state["code"] = format_java_code(st.session_state['code'])

    # col1, col2 = st.sidebar.columns(2)
    # with col1:
    #     format_button = col1.button("Format", on_click=format_code)
    # with col2:
    #     optimize_button = col2.button("Optimized")
        
    # if optimize_button:
    #     st.toast("Success!", icon="✅")


    # st.session_state["converted_code"] = ''    
    # col3, col4 = st.sidebar.columns(2)
    # with col3:
    #     pass
    # with col4:    
        # if st.button("JIRA"):
        #     st.session_state["current_page"] = "Jira"  # Change page to chat page
        #     st.rerun()    
    if st.sidebar.button("View Tickets"):
        st.session_state["current_page"] = "Jira2"  # Change page to chat page
        st.rerun()      
    
    # if st.session_state["converted_code"]!='':
    #         with st.expander("Converted Code", expanded=True):
    #             st.code(st.session_state["converted_code"], language=("Python").lower())
              
    with st.sidebar.popover("feedback"):
        feedback_received_2 = 0
        st.markdown("👋 Hello, Please enter your Feedback here")
        feedback_received = st.text_area("")
        if(st.button("ok")):
            if(feedback_received):
                store_feedback(feedback_received)
                feedback_received_2 = 1
        if(feedback_received_2 == 1):
            st.success("Thank you for your feedback")

# def chat_page():
#     st.title("Chat with Phoenix")

#     if "messages" not in st.session_state:
#         st.session_state["messages"] = [{"role": "Phoenix", "content": "How can I help you?"}]

#     # Displaying the chat history
#     for msg in st.session_state.messages:
#         if msg["role"] == "user":
#             st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
#         else:
#             st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

#     def generate_response():
#         response = ollama.chat(model='llama3.2:1b', stream=True, messages=st.session_state.messages)
#         for partial_resp in response:
#             token = partial_resp["message"]["content"]
#             st.session_state["full_message"] += token
#             yield token

#     if prompt := st.chat_input():
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         st.chat_message("user", avatar="🧑‍💻").write(prompt)
#         st.session_state["full_message"] = ""
#         # st.chat_message("Phoenix", avatar="🤖").write_stream(generate_response)
#         st.chat_message("assistant", avatar="🤖").write_stream(generate_response)
#         st.session_state.messages.append({"role": "Phoenix", "content": st.session_state["full_message"]})

#     if st.sidebar.button("Back"):
#         st.session_state["current_page"] = "first"  # Change page to home
#         st.rerun()  # Trigger rerun to reload the app and navigate to the home page

def chat_page():
    st.title("Chat with Phoenix")

    if "messages" not in st.session_state:
        # Start with a system message or an assistant message.
        st.session_state["messages"] = [{"role": "system", "content": "How can I help you?"}]

    # Displaying the chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
        else:
            st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

    def generate_response():
        response = ollama.chat(model='llama3.2:1b', stream=True, messages=st.session_state.messages)
        for partial_resp in response:
            token = partial_resp["message"]["content"]
            st.session_state["full_message"] += token
            yield token

    if prompt := st.chat_input():
        # Add the user message to the conversation
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar="🧑‍💻").write(prompt)

        # Initialize the response message
        st.session_state["full_message"] = ""
        # Use 'assistant' as the role here, as 'Phoenix' is not a valid role
        st.chat_message("assistant", avatar="🤖").write_stream(generate_response)

        # Once the response is fully generated, add it to the message history
        st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})

    if st.sidebar.button("Back"):
        st.session_state["current_page"] = "first"  # Change page to home
        st.rerun()  # Trigger rerun to reload the app and navigate to the home page

def gen_jira_page():
    st.title("Generate Jira Tickets")
    if st.sidebar.button("Back"):
        st.session_state["current_page"] = "home"  # Change page to home
        st.rerun()  # Trigger rerun to reload the app and navigate to the home page

    # Sample data to represent JIRA tickets
    data = {
        "Ticket ID": ["JIRA-001", "JIRA-002", "JIRA-003", "JIRA-004"],
        "Title": ["Bug in login page", "Add feature X", "Improve performance", "Update documentation"],
        "Assignee": ["Alice", "Bob", "Charlie", "David"],
        "Status": ["To Do", "In Progress", "Review", "Done"],
        "Priority": ["High", "Medium", "Low", "Medium"],
        "Description": [
            "Login page bug causing error 500.",
            "Add new feature X to improve UX.",
            "Optimize database queries for better performance.",
            "Update documentation to reflect recent changes."
        ]
    }

    df = pd.DataFrame(data)
    st.title("JIRA-like Task Dashboard")
    st.sidebar.header("Filter Tasks")

    status_filter = st.sidebar.multiselect("Status", options=df["Status"].unique(), default=df["Status"].unique())
    priority_filter = st.sidebar.multiselect("Priority", options=df["Priority"].unique(), default=df["Priority"].unique())

    filtered_df = df[(df["Status"].isin(status_filter)) & (df["Priority"].isin(priority_filter))]

    st.subheader("Tasks")
    for _, row in filtered_df.iterrows():
        with st.expander(f"{row['Ticket ID']}: {row['Title']}"):
            st.write(f"**Assignee:** {row['Assignee']}")
            st.write(f"**Status:** {row['Status']}")
            st.write(f"**Priority:** {row['Priority']}")
            st.write(f"**Description:** {row['Description']}")

    with st.sidebar.expander("Create New Task"):
        new_title = st.text_input("Task Title")
        new_assignee = st.text_input("Assignee")
        new_status = st.selectbox("Status", options=["To Do", "In Progress", "Review", "Done"])
        new_priority = st.selectbox("Priority", options=["High", "Medium", "Low"])
        new_description = st.text_area("Description")

        if st.button("Add Task"):
            # Add new task to dataframe
            new_task = pd.DataFrame({
                "Ticket ID": [f"JIRA-{len(df)+1:03}"],
                "Title": [new_title],
                "Assignee": [new_assignee],
                "Status": [new_status],
                "Priority": [new_priority],
                "Description": [new_description]
            })
            df = pd.concat([df, new_task], ignore_index=True)
            st.success("Task added successfully!")
            
    st.subheader("Task Overview")
    st.dataframe(df)

def gen_jira_page2():
    st.title("View Jira Tickets")
    
    # Sidebar navigation buttons
    if st.sidebar.button("Back To Home"):
        st.session_state["current_page"] = "home"
        st.rerun()

    if st.sidebar.button("Go To Editor"):
        st.session_state["current_page"] = "first"
        st.rerun()

    # Retrieve all work packages
    work_packages = ps.find_work_packages(proj)

    # Prepare data for the select box
    work_package_options = [
        (wp.id, wp.subject) for wp in work_packages
    ]
    work_package_dict = {f"{wp.id} - {wp.subject}": wp for wp in work_packages}

    # Sidebar: Select a work package
    st.sidebar.title("Select the ticket")
    selected_wp = st.sidebar.selectbox(
        "Select ticket by ID and Subject:",
        options=[f"{wp[0]} - {wp[1]}" for wp in work_package_options],
        help="Select a work package to view or edit details"
    )

    # Fetch selected work package
    selected_work_package = work_package_dict[selected_wp]

    # Display selected work package details
    #st.write(f"### Selected Work Package: {selected_work_package.subject}")
    st.write(f"**ID:** {selected_work_package.id}")
    st.write(f"**Subject:** {selected_work_package.subject}")
    st.write(f"**Created At:** {selected_work_package.createdAt}")
    st.write(f"**Updated At:** {selected_work_package.updatedAt}")

    if 'status' in selected_work_package._links:
        st.write(f"**Status:** {selected_work_package._links['status']['title']}")
    if 'priority' in selected_work_package._links:
        st.write(f"**Priority:** {selected_work_package._links['priority']['title']}")
    if 'author' in selected_work_package._links:
        st.write(f"**Author:** {selected_work_package._links['author']['title']}")

    description = selected_work_package.description.get('raw', 'No description provided')
    st.write(f"**Description:** {description}")

    if 'assignee' in selected_work_package._links and selected_work_package._links['assignee']['href']:
        st.write(f"**Assignee:** {selected_work_package._links['assignee']['title']}")
    else:
        st.write("**Assignee:** Not assigned")

    # Sidebar: Action selection
    action = st.sidebar.selectbox(
        "Select Action",
        options=["View Details", "Update", "Delete"],index=0,
        help="Choose the action you want to perform on a work package"
    )

    # Conditional logic for actions
    if action == "Update":
        # Update interface
        st.subheader("Update ticket")

        # Input fields for updating
        new_status_name = st.selectbox("Select new status", options=status_map.keys())
        new_priority_id = st.selectbox(
            "Select new priority",
            options=list(priority_options.keys()),
            format_func=lambda x: priority_options[x]
        )
        new_assignee_id = st.selectbox(
            "Select new assignee",
            options=list(user_options.keys()),
            format_func=lambda x: user_options[x]
        )
        new_subject = st.text_input(
            "Enter new subject (leave blank to keep current):",
            value=selected_work_package.subject
        )

        # Update button
        if st.button("Update ticket"):
            wp_service = op.get_work_package_service()

            # Update fields
            selected_work_package._links['status'] = {"href": f"/api/v3/statuses/{status_map[new_status_name]}"}
            selected_work_package._links['priority'] = {"href": f"/api/v3/priorities/{new_priority_id}"}
            selected_work_package._links['assignee'] = {"href": f"/api/v3/users/{new_assignee_id}"}
            selected_work_package.subject = new_subject

            # Remove non-updatable fields
            work_package_data = selected_work_package.__dict__
            work_package_data.pop('createdAt', None)
            work_package_data.pop('updatedAt', None)

            # Update the work package
            wp_service.update(selected_work_package)
            st.success(f"Tikcet ID {selected_work_package.id} has been updated.")

    elif action == "Delete":
        # Delete interface
        st.subheader("Delete ticket")

        if st.button("Delete ticket"):
            wp_service = op.get_work_package_service()
            wp_service.delete(selected_work_package)
            st.success(f"Ticket ID {selected_work_package.id} has been deleted.")
            st.rerun()

if st.session_state.get("current_page", "home") == "home":
    main_page()
elif st.session_state["current_page"] == "first":
    first_page()
elif st.session_state["current_page"] == "chat":
    chat_page()
elif st.session_state["current_page"] == "Jira":
    gen_jira_page()
elif st.session_state["current_page"] == "Jira2":
    gen_jira_page2()