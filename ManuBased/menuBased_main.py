import os
import sys
import subprocess
import paramiko
import streamlit as st
from streamlit_chat import message
import google.generativeai as genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
import boto3
from git import Repo
import git
import psutil
import pywhatkit
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from twilio.rest import Client
from googlesearch import search
import instagrapi
import tweepy
import face_recognition
from PIL import Image
import cv2
import speech_recognition as sr
import time
import base64
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Multi-Tool Dashboard",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'linux_creds' not in st.session_state:
    st.session_state.linux_creds = {"ip": "", "username": "", "password": ""}
if 'git_token' not in st.session_state:
    st.session_state.git_token = ""
if 'ssh_client' not in st.session_state:
    st.session_state.ssh_client = None
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None
if 'gemini_chat' not in st.session_state:
    st.session_state.gemini_chat = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'aws_creds' not in st.session_state:
    st.session_state.aws_creds = {"access_key": "", "secret_key": "", "region": ""}
if 'voice_input' not in st.session_state:
    st.session_state.voice_input = ""

# Common Linux Commands
LINUX_COMMANDS = {
    "File Operations": {
        "ls": "List directory contents",
        "cd": "Change directory",
        "pwd": "Print working directory",
        "cp": "Copy files/directories",
        "mv": "Move/rename files/directories",
        "rm": "Remove files/directories",
        "mkdir": "Create directory",
        "touch": "Create empty file",
    },
    "System Info": {
        "uname -a": "Show system information",
        "df -h": "Show disk space usage",
        "free -h": "Show memory usage",
        "top": "Display running processes",
        "htop": "Interactive process viewer",
        "ps aux": "Show all running processes",
    },
    "Networking": {
        "ifconfig": "Network interface configuration",
        "ping": "Test network connectivity",
        "netstat": "Network statistics",
        "ssh": "Secure shell remote login",
        "scp": "Secure copy between hosts",
        "wget": "Download files from web",
        "curl": "Transfer data from/to server",
    },
    "Package Management": {
        "apt update": "Update package list (Debian/Ubuntu)",
        "apt upgrade": "Upgrade packages (Debian/Ubuntu)",
        "apt install <package>": "Install package (Debian/Ubuntu)",
        "yum update": "Update packages (RHEL/CentOS)",
        "yum install <package>": "Install package (RHEL/CentOS)",
        "dnf install <package>": "Install package (Fedora)",
    },
    "Permissions": {
        "chmod": "Change file permissions",
        "chown": "Change file owner",
        "chgrp": "Change file group",
    },
    "Process Management": {
        "kill": "Terminate process by PID",
        "killall": "Terminate processes by name",
        "pkill": "Terminate process by name",
        "bg": "Run process in background",
        "fg": "Bring process to foreground",
        "jobs": "List background processes",
    },
    "Text Processing": {
        "grep": "Search text using patterns",
        "awk": "Text processing language",
        "sed": "Stream editor",
        "cat": "Concatenate and display files",
        "less": "View file contents",
        "head": "Show first lines of file",
        "tail": "Show last lines of file",
    },
    "Compression": {
        "tar": "Tape archive utility",
        "gzip": "Compress files",
        "gunzip": "Decompress files",
        "zip": "Package and compress files",
        "unzip": "Decompress zip files",
    },
    "Users/Groups": {
        "useradd": "Add user",
        "userdel": "Delete user",
        "passwd": "Change password",
        "groupadd": "Add group",
        "usermod": "Modify user",
    },
    "Miscellaneous": {
        "history": "Show command history",
        "man": "Display manual pages",
        "alias": "Create command alias",
        "echo": "Display message",
        "date": "Display/set system date",
        "cal": "Display calendar",
    }
}

# Initialize Gemini AI
def init_gemini():
    if st.session_state.gemini_api_key:
        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            st.session_state.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            st.success("Gemini AI initialized successfully!")
            return True
        except Exception as e:
            st.error(f"Error initializing Gemini: {str(e)}")
            return False
    else:
        st.warning("Please enter Gemini API Key first")
        return False

# SSH Connection Functions
def connect_ssh():
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            st.session_state.linux_creds["ip"],
            username=st.session_state.linux_creds["username"],
            password=st.session_state.linux_creds["password"]
        )
        st.session_state.ssh_client = client
        st.success("SSH Connection Established Successfully!")
        return True
    except Exception as e:
        st.error(f"SSH Connection Failed: {str(e)}")
        return False

def disconnect_ssh():
    if st.session_state.ssh_client:
        st.session_state.ssh_client.close()
        st.session_state.ssh_client = None
        st.success("SSH Connection Closed")

def execute_ssh_command(command):
    if not st.session_state.ssh_client:
        st.warning("SSH Connection not established")
        return None
    
    try:
        stdin, stdout, stderr = st.session_state.ssh_client.exec_command(command)
        output = stdout.read().decode()
        error = stderr.read().decode()
        
        if error:
            return f"Error: {error}"
        return output
    except Exception as e:
        return f"Exception: {str(e)}"

# Voice Recognition
def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Speak now")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            st.session_state.voice_input = text
            st.write(f"You said: {text}")
            return text
        except Exception as e:
            st.error(f"Error recognizing speech: {str(e)}")
            return ""

# Home Page
def home_page():
    st.title("🚀 Multi-Tool Dashboard")
    st.markdown("""
    Welcome to the Multi-Tool Dashboard! This application integrates various functionalities including:
    
    - **Linux SSH Operations** - Connect to remote Linux systems and execute commands
    - **Gemini AI Integration** - Get AI assistance for your technical queries
    - **Git Operations** - Manage repositories and collaborate on projects
    - **AWS Cloud Operations** - Work with EC2, S3, and CloudWatch
    - **Python Utilities** - Various Python automation tasks
    - **Machine Learning** - Data imputation and analysis
    - **Voice Recognition** - Control the app with your voice
    
    Use the sidebar to navigate between different sections and configure your credentials.
    """)
    
    st.image("https://cdn.pixabay.com/photo/2017/06/20/22/14/man-2425121_1280.jpg", use_column_width=True)
    
    if st.button("Use Voice Command"):
        voice_input()
        if st.session_state.voice_input:
            process_voice_command(st.session_state.voice_input)

def process_voice_command(command):
    command = command.lower()
    if "linux" in command:
        st.session_state.current_page = "Linux"
    elif "git" in command or "github" in command:
        st.session_state.current_page = "Git"
    elif "aws" in command or "amazon" in command:
        st.session_state.current_page = "AWS"
    elif "python" in command:
        st.session_state.current_page = "Python"
    elif "machine learning" in command or "ml" in command:
        st.session_state.current_page = "ML"
    elif "home" in command:
        st.session_state.current_page = "Home"
    else:
        st.warning("Command not recognized. Please try again.")

# Credentials Page
def credentials_page():
    st.title("🔑 Credentials Configuration")
    
    with st.expander("Gemini API Configuration"):
        st.session_state.gemini_api_key = st.text_input("Enter Gemini API Key", 
                                                      value=st.session_state.gemini_api_key, 
                                                      type="password")
        if st.button("Initialize Gemini"):
            init_gemini()
    
    with st.expander("Linux SSH Credentials"):
        st.session_state.linux_creds["ip"] = st.text_input("Linux Server IP", 
                                                         value=st.session_state.linux_creds["ip"])
        st.session_state.linux_creds["username"] = st.text_input("Username", 
                                                               value=st.session_state.linux_creds["username"])
        st.session_state.linux_creds["password"] = st.text_input("Password", 
                                                               value=st.session_state.linux_creds["password"], 
                                                               type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Connect to Linux Server"):
                connect_ssh()
        with col2:
            if st.button("Disconnect"):
                disconnect_ssh()
    
    with st.expander("Git Configuration"):
        st.session_state.git_token = st.text_input("Git Personal Access Token", 
                                                 value=st.session_state.git_token, 
                                                 type="password")
    
    with st.expander("AWS Credentials"):
        st.session_state.aws_creds["access_key"] = st.text_input("AWS Access Key", 
                                                               value=st.session_state.aws_creds["access_key"], 
                                                               type="password")
        st.session_state.aws_creds["secret_key"] = st.text_input("AWS Secret Key", 
                                                               value=st.session_state.aws_creds["secret_key"], 
                                                               type="password")
        st.session_state.aws_creds["region"] = st.text_input("AWS Region", 
                                                           value=st.session_state.aws_creds["region"])

# Linux Page
def linux_page():
    st.title("🐧 Linux Operations")
    
    if not st.session_state.ssh_client:
        st.warning("Please connect to a Linux server from the Credentials page first")
        return
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Blog on Linux Companies", 
        "GUI Program Commands", 
        "Change Program Logo", 
        "Add Terminals/GUI", 
        "Send Communications", 
        "Command Explorer"
    ])
    
    with tab1:
        st.header("📝 Blog on Companies Using Linux")
        if st.button("Generate Blog Post"):
            if st.session_state.gemini_model:
                prompt = """Write a detailed blog post about companies using Linux, explaining:
                1. Why they choose Linux over other operating systems
                2. What specific benefits they gain from using Linux
                3. Examples of major companies using Linux and their use cases
                4. Cost savings and performance improvements
                5. Future trends of Linux in enterprise environments"""
                
                with st.spinner("Generating blog post..."):
                    response = st.session_state.gemini_model.generate_content(prompt)
                    st.markdown(response.text)
            else:
                st.warning("Gemini AI not initialized")
    
    with tab2:
        st.header("🖥️ GUI Program Commands")
        gui_programs = st.multiselect(
            "Select GUI Programs to Analyze",
            ["Nautilus (File Manager)", "Gedit (Text Editor)", "LibreOffice", "Firefox", "GNOME Terminal"]
        )
        
        if st.button("Find Underlying Commands"):
            if st.session_state.gemini_model:
                prompt = f"""For the following Linux GUI programs: {', '.join(gui_programs)}, 
                provide the underlying terminal commands that these applications use, 
                explain what each command does, and show how you could perform similar 
                operations directly from the terminal."""
                
                with st.spinner("Analyzing GUI programs..."):
                    response = st.session_state.gemini_model.generate_content(prompt)
                    st.markdown(response.text)
            else:
                st.warning("Gemini AI not initialized")
    
    with tab3:
        st.header("🖼️ Change Program Logo/Icon")
        program = st.text_input("Enter program name to change icon (e.g., firefox)")
        icon_file = st.file_uploader("Upload new icon file (.png or .svg)", type=["png", "svg"])
        
        if st.button("Change Icon"):
            if program and icon_file:
                # Save the uploaded file temporarily
                with open(f"/tmp/{icon_file.name}", "wb") as f:
                    f.write(icon_file.getbuffer())
                
                # Command to change icon
                commands = [
                    f"sudo cp /tmp/{icon_file.name} /usr/share/icons/hicolor/48x48/apps/{program}.png",
                    f"sudo cp /tmp/{icon_file.name} /usr/share/icons/hicolor/48x48/apps/{program}.svg",
                    f"sudo update-icon-caches /usr/share/icons/*"
                ]
                
                results = []
                for cmd in commands:
                    results.append(f"$ {cmd}")
                    output = execute_ssh_command(cmd)
                    if output:
                        results.append(output)
                
                st.code("\n".join(results))
                st.success("Icon changed successfully! You may need to restart the application.")
            else:
                st.warning("Please provide both program name and icon file")
    
    with tab4:
        st.header("➕ Add Terminals/GUI Interfaces")
        option = st.selectbox("Select enhancement", [
            "Install new terminal emulator",
            "Add GUI interface to headless server",
            "Install desktop environment",
            "Customize existing terminal"
        ])
        
        if st.button("Implement Change"):
            commands = []
            if option == "Install new terminal emulator":
                commands = [
                    "sudo apt update",
                    "sudo apt install tilix -y",
                    "echo 'Tilix terminal installed successfully'"
                ]
            elif option == "Add GUI interface to headless server":
                commands = [
                    "sudo apt update",
                    "sudo apt install xorg xfce4 -y",
                    "echo 'XFCE4 GUI installed. Start with startxfce4 command'"
                ]
            elif option == "Install desktop environment":
                desktop = st.selectbox("Select desktop environment", ["GNOME", "KDE", "XFCE", "LXDE"])
                pkg = {"GNOME": "ubuntu-gnome-desktop", "KDE": "kubuntu-desktop", "XFCE": "xfce4", "LXDE": "lxde"}[desktop]
                commands = [
                    "sudo apt update",
                    f"sudo apt install {pkg} -y",
                    f"echo '{desktop} desktop environment installed'"
                ]
            elif option == "Customize existing terminal":
                commands = [
                    "echo 'Customizing terminal...'",
                    "sudo apt install zsh -y",
                    "sh -c '$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)'",
                    "echo 'Oh My Zsh installed for terminal customization'"
                ]
            
            results = []
            for cmd in commands:
                results.append(f"$ {cmd}")
                output = execute_ssh_command(cmd)
                if output:
                    results.append(output)
            
            st.code("\n".join(results))
    
    with tab5:
        st.header("📨 Send Communications from Terminal")
        comm_type = st.selectbox("Select communication type", [
            "Email", "WhatsApp", "Twitter", "SMS"
        ])
        
        if comm_type == "Email":
            recipient = st.text_input("Recipient email")
            subject = st.text_input("Subject")
            body = st.text_area("Message body")
            
            if st.button("Send Email"):
                commands = [
                    f"echo '{body}' | mail -s '{subject}' {recipient}",
                    "echo 'Email sent using mailutils package'"
                ]
                
                results = []
                for cmd in commands:
                    results.append(f"$ {cmd}")
                    output = execute_ssh_command(cmd)
                    if output:
                        results.append(output)
                
                st.code("\n".join(results))
        
        elif comm_type == "WhatsApp":
            st.warning("WhatsApp requires external services. This is a simulation.")
            number = st.text_input("Phone number (with country code)")
            message = st.text_area("Message")
            
            if st.button("Send WhatsApp Message"):
                st.info("On a real system, you would use tools like whatsapp-web.js or Twilio API")
                st.code(f"""
                # Example using yowsup (WhatsApp CLI client)
                yowsup-cli demos -c config.example -s {number} "{message}"
                """)
        
        elif comm_type == "Twitter":
            message = st.text_area("Tweet content", max_chars=280)
            
            if st.button("Post Tweet"):
                st.info("On a real system, you would use Twitter API with twurl or tweepy")
                st.code(f"""
                # Example using twurl (Twitter CLI client)
                twurl -d 'status={message}' /1.1/statuses/update.json
                """)
        
        elif comm_type == "SMS":
            number = st.text_input("Phone number (with country code)")
            message = st.text_area("Message")
            
            if st.button("Send SMS"):
                st.info("On a real system, you would use services like Twilio or AWS SNS")
                st.code(f"""
                # Example using AWS SNS CLI
                aws sns publish --phone-number {number} --message "{message}"
                """)
    
    with tab6:
        st.header("🔍 Linux Command Explorer")
        
        # Voice command for Linux
        if st.button("Voice Command for Linux"):
            cmd = voice_input()
            if cmd:
                st.write(f"Executing: {cmd}")
                output = execute_ssh_command(cmd)
                st.code(output)
        
        # Display categorized commands
        for category, commands in LINUX_COMMANDS.items():
            with st.expander(f"{category} Commands"):
                for cmd, desc in commands.items():
                    st.code(f"{cmd} - {desc}")
        
        # Custom command execution
        custom_cmd = st.text_input("Enter custom Linux command")
        if st.button("Execute Command"):
            if custom_cmd:
                output = execute_ssh_command(custom_cmd)
                st.code(output)
            else:
                st.warning("Please enter a command")
        
        # Ask Gemini about commands
        if st.session_state.gemini_model:
            command_question = st.text_input("Ask Gemini about a Linux command")
            if command_question:
                prompt = f"Explain this Linux command in detail with examples: {command_question}"
                with st.spinner("Asking Gemini..."):
                    response = st.session_state.gemini_model.generate_content(prompt)
                    st.markdown(response.text)

# ML Page
def ml_page():
    st.title("🤖 Machine Learning Operations")
    
    st.markdown("""
    ## Missing Value Imputation Techniques
    
    This section demonstrates various techniques for handling missing values in datasets,
    with a focus on determining if Linear Regression is appropriate for imputing missing target values.
    """)
    
    # Sample dataset
    data = {
        'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'X2': [5, 4, 3, 2, 1, 5, 4, 3, 2, 1],
        'Y': [10, 8, 6, 4, 2, None, 8, 6, None, 2]
    }
    df = pd.DataFrame(data)
    
    st.subheader("Original Dataset with Missing Values")
    st.dataframe(df)
    
    st.subheader("1. Basic Imputation Techniques")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Mean Imputation**")
        mean_df = df.copy()
        mean_df['Y'] = mean_df['Y'].fillna(mean_df['Y'].mean())
        st.dataframe(mean_df)
    
    with col2:
        st.markdown("**Median Imputation**")
        median_df = df.copy()
        median_df['Y'] = median_df['Y'].fillna(median_df['Y'].median())
        st.dataframe(median_df)
    
    with col3:
        st.markdown("**Mode Imputation**")
        mode_df = df.copy()
        mode_df['Y'] = mode_df['Y'].fillna(mode_df['Y'].mode()[0])
        st.dataframe(mode_df)
    
    st.subheader("2. Advanced Imputation Techniques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**KNN Imputation**")
        knn_df = df.copy()
        imputer = KNNImputer(n_neighbors=2)
        knn_imputed = imputer.fit_transform(knn_df)
        knn_df = pd.DataFrame(knn_imputed, columns=df.columns)
        st.dataframe(knn_df)
    
    with col2:
        st.markdown("**Linear Regression Imputation**")
        st.markdown("""
        **Is Linear Regression appropriate?**
        
        Linear Regression can be appropriate for imputing missing target values when:
        - There is a linear relationship between features and target
        - The missingness is random (MAR or MCAR)
        - The dataset is not too small
        """)
        
        if st.button("Perform Linear Regression Imputation"):
            # Prepare data
            train_df = df.dropna()
            test_df = df[df['Y'].isna()]
            
            if len(train_df) > 1:
                X_train = train_df[['X1', 'X2']]
                y_train = train_df['Y']
                
                # Train model
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Predict missing values
                X_test = test_df[['X1', 'X2']]
                y_pred = model.predict(X_test)
                
                # Fill missing values
                lr_df = df.copy()
                lr_df.loc[lr_df['Y'].isna(), 'Y'] = y_pred
                
                st.dataframe(lr_df)
                
                # Show model coefficients
                st.markdown("**Model Coefficients:**")
                st.write(f"Intercept: {model.intercept_:.2f}")
                st.write(f"X1 Coefficient: {model.coef_[0]:.2f}")
                st.write(f"X2 Coefficient: {model.coef_[1]:.2f}")
                
                # Show scatter plot
                fig, ax = plt.subplots()
                ax.scatter(train_df['X1'], train_df['Y'], color='blue', label='Actual')
                ax.scatter(test_df['X1'], y_pred, color='red', label='Predicted (Imputed)')
                ax.set_xlabel('X1')
                ax.set_ylabel('Y')
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("Not enough data to train the model")

# AWS Page
def aws_page():
    st.title("☁️ AWS Operations")
    
    if not st.session_state.aws_creds["access_key"]:
        st.warning("Please configure AWS credentials in the Credentials page")
        return
    
    # Initialize boto3 client
    try:
        ec2 = boto3.client(
            'ec2',
            aws_access_key_id=st.session_state.aws_creds["access_key"],
            aws_secret_access_key=st.session_state.aws_creds["secret_key"],
            region_name=st.session_state.aws_creds["region"]
        )
        st.session_state.ec2_client = ec2
    except Exception as e:
        st.error(f"Failed to initialize AWS client: {str(e)}")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "AWS Use Case Blog", 
        "EC2 Operations", 
        "CloudWatch Logs", 
        "S3 Storage Classes"
    ])
    
    with tab1:
        st.header("📝 AWS Use Case Study")
        if st.button("Generate Blog Post"):
            if st.session_state.gemini_model:
                prompt = """Write a detailed blog post about AWS use case studies, covering:
                1. Different industries using AWS and their specific use cases
                2. Architecture diagrams of typical AWS implementations
                3. Cost savings and performance benefits achieved
                4. Lessons learned from real-world implementations
                5. Future trends in cloud computing with AWS"""
                
                with st.spinner("Generating blog post..."):
                    response = st.session_state.gemini_model.generate_content(prompt)
                    st.markdown(response.text)
            else:
                st.warning("Gemini AI not initialized")
    
    with tab2:
        st.header("🖥️ EC2 Instance Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Launch EC2 Instance")
            ami_id = st.text_input("AMI ID", "ami-0c55b159cbfafe1f0")
            instance_type = st.selectbox("Instance Type", ["t2.micro", "t2.small", "t2.medium"])
            
            if st.button("Launch Instance"):
                try:
                    response = st.session_state.ec2_client.run_instances(
                        ImageId=ami_id,
                        InstanceType=instance_type,
                        MinCount=1,
                        MaxCount=1,
                    )
                    instance_id = response['Instances'][0]['InstanceId']
                    st.success(f"Instance {instance_id} launched successfully!")
                    st.json(response)
                except Exception as e:
                    st.error(f"Error launching instance kill process or rdelay for sometimes: {str(e)}")
        
        with col2:
            st.subheader("Terminate EC2 Instance")
            instances = st.session_state.ec2_client.describe_instances()
            instance_ids = []
            for reservation in instances['Reservations']:
                for instance in reservation['Instances']:
                    instance_ids.append(instance['InstanceId'])
            
            selected_instance = st.selectbox("Select instance to terminate", instance_ids)
            
            if st.button("Terminate Instance"):
                try:
                    response = st.session_state.ec2_client.terminate_instances(
                        InstanceIds=[selected_instance]
                    )
                    st.success(f"Instance {selected_instance} termination initiated!")
                    st.json(response)
                except Exception as e:
                    st.error(f"Error terminating instance: {str(e)}")
        
        st.subheader("Boto3 Code Example")
        st.code("""
        # Launch EC2 instance
        import boto3
        
        ec2 = boto3.client('ec2',
                          aws_access_key_id='YOUR_ACCESS_KEY',
                          aws_secret_access_key='YOUR_SECRET_KEY',
                          region_name='us-east-1')
        
        response = ec2.run_instances(
            ImageId='ami-0c55b159cbfafe1f0',
            InstanceType='t2.micro',
            MinCount=1,
            MaxCount=1
        )
        
        print(response)
        """)
    
    with tab3:
        st.header("📜 CloudWatch Logs Access")
        log_group = st.text_input("Log Group Name", "/var/log/syslog")
        
        if st.button("Get Log Events"):
            try:
                logs = boto3.client(
                    'logs',
                    aws_access_key_id=st.session_state.aws_creds["access_key"],
                    aws_secret_access_key=st.session_state.aws_creds["secret_key"],
                    region_name=st.session_state.aws_creds["region"]
                )
                
                log_streams = logs.describe_log_streams(
                    logGroupName=log_group,
                    orderBy='LastEventTime',
                    descending=True,
                    limit=5
                )
                
                st.subheader("Recent Log Streams")
                for stream in log_streams['logStreams']:
                    st.write(f"Stream: {stream['logStreamName']}")
                    st.write(f"Last Event: {stream['lastEventTimestamp']}")
                    
                    events = logs.get_log_events(
                        logGroupName=log_group,
                        logStreamName=stream['logStreamName'],
                        limit=10
                    )
                    
                    for event in events['events']:
                        st.code(event['message'])
                
                st.subheader("Boto3 Code Example")
                st.code("""
                # Access CloudWatch Logs
                import boto3
                
                logs = boto3.client('logs',
                                  aws_access_key_id='YOUR_ACCESS_KEY',
                                  aws_secret_access_key='YOUR_SECRET_KEY',
                                  region_name='us-east-1')
                
                # Get log streams
                streams = logs.describe_log_streams(
                    logGroupName='/var/log/syslog',
                    orderBy='LastEventTime',
                    descending=True,
                    limit=5
                )
                
                # Get log events
                events = logs.get_log_events(
                    logGroupName='/var/log/syslog',
                    logStreamName=streams['logStreams'][0]['logStreamName'],
                    limit=10
                )
                
                for event in events['events']:
                    print(event['message'])
                """)
            except Exception as e:
                st.error(f"Error accessing logs: {str(e)}")
    
    with tab4:
        st.header("🗄️ S3 Storage Classes")
        if st.button("Generate Blog Post on S3 Storage Classes"):
            if st.session_state.gemini_model:
                prompt = """Write a detailed blog post about AWS S3 storage classes covering:
                1. Comparison of all storage classes (Standard, Intelligent-Tiering, 
                   Standard-IA, One Zone-IA, Glacier, Glacier Deep Archive)
                2. Cost differences between storage classes
                3. Performance characteristics of each class
                4. Ideal use cases for each storage class
                5. Lifecycle policies and transition strategies
                6. Recent updates and new features"""
                
                with st.spinner("Generating blog post..."):
                    response = st.session_state.gemini_model.generate_content(prompt)
                    st.markdown(response.text)
            else:
                st.warning("Gemini AI not initialized")

# Git Page
def git_page():
    st.title("🐙 Git Operations")
    
    if not st.session_state.git_token:
        st.warning("Please configure Git token in the Credentials page")
        return
    
    tab1, tab2 = st.tabs(["Repository Management", "Open Source Contribution"])
    
    with tab1:
        st.header("🔄 Repository Management")
        
        repo_name = st.text_input("New Repository Name")
        folder_path = st.text_input("Local Folder Path", "./my_project")
        file_content = st.text_area("File Content", "print('Hello World!')")
        commit_msg = st.text_input("Commit Message", "Initial commit")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Create Repository"):
                try:
                    # Create folder
                    os.makedirs(folder_path, exist_ok=True)
                    
                    # Initialize repo
                    repo = Repo.init(folder_path)
                    
                    # Create file
                    with open(os.path.join(folder_path, "main.py"), "w") as f:
                        f.write(file_content)
                    
                    # Add and commit
                    repo.index.add(["main.py"])
                    repo.index.commit(commit_msg)
                    
                    st.success(f"Local repository created at {folder_path}")
                except Exception as e:
                    st.error(f"Error creating repository: {str(e)}")
        
        with col2:
            if st.button("Push to GitHub"):
                try:
                    repo = Repo(folder_path)
                    
                    # Create remote
                    origin = repo.create_remote(
                        "origin", 
                        f"https://{st.session_state.git_token}@github.com/yourusername/{repo_name}.git"
                    )
                    
                    # Push
                    origin.push(refspec='HEAD:refs/heads/main')
                    
                    st.success(f"Repository pushed to GitHub: {repo_name}")
                except Exception as e:
                    st.error(f"Error pushing to GitHub: {str(e)}")
        
        st.subheader("Branch Management")
        branch_name = st.text_input("New Branch Name", "feature1")
        
        if st.button("Create and Merge Branch"):
            try:
                repo = Repo(folder_path)
                
                # Create branch
                new_branch = repo.create_head(branch_name)
                new_branch.checkout()
                
                # Make changes
                with open(os.path.join(folder_path, "feature.py"), "w") as f:
                    f.write("print('New feature!')")
                
                # Commit in branch
                repo.index.add(["feature.py"])
                repo.index.commit(f"Add {branch_name} feature")
                
                # Switch to main and merge
                repo.heads.main.checkout()
                repo.git.merge(branch_name)
                
                st.success(f"Branch {branch_name} created and merged successfully!")
            except Exception as e:
                st.error(f"Error with branch operations: {str(e)}")
    
    with tab2:
        st.header("🤝 Open Source Contribution")
        
        repo_url = st.text_input("Repository URL to Fork", "https://github.com/username/repository")
        local_path = st.text_input("Local Clone Path", "./cloned_repo")
        
        if st.button("Fork and Clone"):
            st.info("""
            Note: Actual forking must be done on GitHub first through the UI or API.
            This will clone an existing repository you have access to.
            """)
            
            try:
                # Clone the repo
                repo = Repo.clone_from(
                    repo_url,
                    local_path
                )
                
                st.success(f"Repository cloned to {local_path}")
            except Exception as e:
                st.error(f"Error cloning repository: {str(e)}")
        
        change_description = st.text_area("Changes Description")
        pr_title = st.text_input("Pull Request Title")
        pr_body = st.text_area("Pull Request Description")
        
        if st.button("Create Pull Request"):
            st.info("""
            Note: Actual PR creation requires GitHub API calls or UI interaction.
            Here's the typical workflow:
            1. Fork the repo on GitHub
            2. Clone your fork locally
            3. Create a new branch and make changes
            4. Push changes to your fork
            5. Create PR from GitHub UI
            """)
            
            st.code(f"""
            # Example GitHub CLI command to create PR
            gh pr create --title "{pr_title}" --body "{pr_body}" --head yourusername:feature-branch --base main
            """)

# Python Page
def python_page():
    st.title("🐍 Python Utilities")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "System Info", 
        "Communication", 
        "Web Automation", 
        "Image Processing", 
        "Data Structures", 
        "Voice Control"
    ])
    
    with tab1:
        st.header("💻 System Information")
        
        if st.button("Read RAM Usage"):
            ram = psutil.virtual_memory()
            st.write(f"Total: {ram.total / (1024**3):.2f} GB")
            st.write(f"Available: {ram.available / (1024**3):.2f} GB")
            st.write(f"Used: {ram.used / (1024**3):.2f} GB")
            st.write(f"Percentage: {ram.percent}%")
            
            st.code("""
            import psutil
            
            ram = psutil.virtual_memory()
            print(f"Total RAM: {ram.total / (1024**3):.2f} GB")
            print(f"Used RAM: {ram.used / (1024**3):.2f} GB")
            """)
    
    with tab2:
        st.header("📨 Communication Tools")
        
        comm_type = st.selectbox("Select communication method", [
            "WhatsApp Message", "Email", "SMS", "Phone Call"
        ])
        
        if comm_type == "WhatsApp Message":
            number = st.text_input("Phone number with country code")
            message = st.text_area("Message")
            hour = st.number_input("Hour (24-hour format)", min_value=0, max_value=23, value=12)
            minute = st.number_input("Minute", min_value=0, max_value=59, value=0)
            
            if st.button("Send WhatsApp Message"):
                try:
                    pywhatkit.sendwhatmsg(f"+{number}", message, hour, minute)
                    st.success("Message scheduled to be sent!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                
                st.code("""
                import pywhatkit
                
                # Send WhatsApp message
                pywhatkit.sendwhatmsg("+1234567890", "Hello from Python!", 12, 0)
                """)
        
        elif comm_type == "Email":
            sender = st.text_input("Your email")
            password = st.text_input("Password", type="password")
            receiver = st.text_input("Recipient email")
            subject = st.text_input("Subject")
            body = st.text_area("Message body")
            
            if st.button("Send Email"):
                try:
                    msg = MIMEMultipart()
                    msg['From'] = sender
                    msg['To'] = receiver
                    msg['Subject'] = subject
                    msg.attach(MIMEText(body, 'plain'))
                    
                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.starttls()
                    server.login(sender, password)
                    text = msg.as_string()
                    server.sendmail(sender, receiver, text)
                    server.quit()
                    
                    st.success("Email sent successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                
                st.code("""
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                
                msg = MIMEMultipart()
                msg['From'] = 'your@gmail.com'
                msg['To'] = 'recipient@example.com'
                msg['Subject'] = 'Subject here'
                msg.attach(MIMEText('Message body', 'plain'))
                
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login('your@gmail.com', 'password')
                server.sendmail('your@gmail.com', 'recipient@example.com', msg.as_string())
                server.quit()
                """)
        
        elif comm_type == "SMS":
            st.warning("Requires Twilio account and credentials")
            account_sid = st.text_input("Twilio Account SID")
            auth_token = st.text_input("Twilio Auth Token", type="password")
            twilio_number = st.text_input("Twilio Phone Number")
            to_number = st.text_input("Recipient Phone Number")
            message = st.text_area("Message")
            
            if st.button("Send SMS"):
                try:
                    client = Client(account_sid, auth_token)
                    message = client.messages.create(
                        body=message,
                        from_=twilio_number,
                        to=to_number
                    )
                    st.success(f"SMS sent! SID: {message.sid}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                
                st.code("""
                from twilio.rest import Client
                
                account_sid = 'your_account_sid'
                auth_token = 'your_auth_token'
                client = Client(account_sid, auth_token)
                
                message = client.messages.create(
                    body='Hello from Python!',
                    from_='+1234567890',
                    to='+0987654321'
                )
                
                print(message.sid)
                """)
        
        elif comm_type == "Phone Call":
            st.warning("Requires Twilio account and credentials")
            account_sid = st.text_input("Twilio Account SID")
            auth_token = st.text_input("Twilio Auth Token", type="password")
            twilio_number = st.text_input("Twilio Phone Number")
            to_number = st.text_input("Recipient Phone Number")
            message = st.text_area("Message to speak")
            
            if st.button("Make Phone Call"):
                try:
                    client = Client(account_sid, auth_token)
                    call = client.calls.create(
                        twiml=f'<Response><Say>{message}</Say></Response>',
                        from_=twilio_number,
                        to=to_number
                    )
                    st.success(f"Call initiated! SID: {call.sid}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                
                st.code("""
                from twilio.rest import Client
                
                account_sid = 'your_account_sid'
                auth_token = 'your_auth_token'
                client = Client(account_sid, auth_token)
                
                call = client.calls.create(
                    twiml='<Response><Say>Hello from Python!</Say></Response>',
                    from_='+1234567890',
                    to='+0987654321'
                )
                
                print(call.sid)
                """)
    
    with tab3:
        st.header("🌐 Web Automation")
        
        web_task = st.selectbox("Select web task", [
            "Google Search", 
            "Social Media Post", 
            "Website Scraping"
        ])
        
        if web_task == "Google Search":
            query = st.text_input("Search Query")
            num_results = st.number_input("Number of results", min_value=1, max_value=10, value=3)
            
            if st.button("Search"):
                try:
                    results = []
                    for j in search(query, num_results=num_results):
                        results.append(j)
                    
                    st.write("Search Results:")
                    for i, result in enumerate(results, 1):
                        st.write(f"{i}. {result}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                
                st.code("""
                from googlesearch import search
                
                query = "Python programming"
                for result in search(query, num_results=3):
                    print(result)
                """)
        
        elif web_task == "Social Media Post":
            platform = st.selectbox("Select platform", ["Twitter", "Instagram", "Facebook"])
            message = st.text_area("Post Content")
            image = st.file_uploader("Upload image (optional)", type=["jpg", "png"])
            
            if st.button(f"Post to {platform}"):
                st.info("""
                Note: Actual posting requires API keys and authentication.
                Here's sample code for Twitter using Tweepy:
                """)
                
                st.code(f"""
                import tweepy
                
                # Authenticate
                auth = tweepy.OAuth1UserHandler(
                    "API_KEY", "API_SECRET",
                    "ACCESS_TOKEN", "ACCESS_TOKEN_SECRET"
                )
                api = tweepy.API(auth)
                
                # Post tweet
                api.update_status("{message}")
                
                # Post with image
                if image:
                    api.update_status_with_media("{message}", "image.jpg")
                """)
        
        elif web_task == "Website Scraping":
            url = st.text_input("Website URL")
            
            if st.button("Scrape Website"):
                st.info("""
                Note: Actual scraping depends on website structure.
                Here's sample code using BeautifulSoup:
                """)
                
                st.code(f"""
                import requests
                from bs4 import BeautifulSoup
                
                url = "{url}"
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Example: Get all links
                for link in soup.find_all('a'):
                    print(link.get('href'))
                
                # Example: Get all text
                print(soup.get_text())
                """)
    
    with tab4:
        st.header("🖼️ Image Processing")
        
        img_task = st.selectbox("Select image task", [
            "Create Digital Image", 
            "Face Swapping", 
            "Image Filters"
        ])
        
        if img_task == "Create Digital Image":
            width = st.number_input("Width", min_value=100, max_value=1000, value=400)
            height = st.number_input("Height", min_value=100, max_value=1000, value=300)
            color = st.color_picker("Background Color", "#ff0000")
            text = st.text_input("Text to add")
            text_color = st.color_picker("Text Color", "#ffffff")
            
            if st.button("Generate Image"):
                try:
                    from PIL import Image, ImageDraw, ImageFont
                    
                    # Create image
                    img = Image.new("RGB", (width, height), color)
                    draw = ImageDraw.Draw(img)
                    
                    # Add text
                    font = ImageFont.load_default()
                    text_width, text_height = draw.textsize(text, font)
                    x = (width - text_width) / 2
                    y = (height - text_height) / 2
                    draw.text((x, y), text, fill=text_color, font=font)
                    
                    # Display
                    st.image(img, caption="Generated Image")
                    
                    # Download option
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="Download Image",
                        data=byte_im,
                        file_name="generated_image.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                
                st.code("""
                from PIL import Image, ImageDraw, ImageFont
                
                # Create image
                img = Image.new("RGB", (400, 300), "#ff0000")
                draw = ImageDraw.Draw(img)
                
                # Add text
                font = ImageFont.load_default()
                draw.text((150, 140), "Hello World!", fill="#ffffff", font=font)
                
                # Save
                img.save("generated_image.png")
                """)
        
        elif img_task == "Face Swapping":
            st.warning("This requires face_recognition library and dlib")
            image1 = st.file_uploader("Upload first image", type=["jpg", "png"])
            image2 = st.file_uploader("Upload second image", type=["jpg", "png"])
            
            if st.button("Swap Faces") and image1 and image2:
                try:
                    # Load images
                    img1 = face_recognition.load_image_file(image1)
                    img2 = face_recognition.load_image_file(image2)
                    
                    # Find face locations
                    face_locations1 = face_recognition.face_locations(img1)
                    face_locations2 = face_recognition.face_locations(img2)
                    
                    if len(face_locations1) == 0 or len(face_locations2) == 0:
                        st.warning("No faces detected in one or both images")
                        return
                    
                    # Get face encodings
                    face_encoding1 = face_recognition.face_encodings(img1, [face_locations1[0]])[0]
                    face_encoding2 = face_recognition.face_encodings(img2, [face_locations2[0]])[0]
                    
                    # Convert to OpenCV format
                    img1_cv = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
                    img2_cv = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
                    
                    # Swap faces
                    top1, right1, bottom1, left1 = face_locations1[0]
                    top2, right2, bottom2, left2 = face_locations2[0]
                    
                    face1 = img1_cv[top1:bottom1, left1:right1]
                    face2 = img2_cv[top2:bottom2, left2:right2]
                    
                    # Resize faces to match
                    face1 = cv2.resize(face1, (right2-left2, bottom2-top2))
                    face2 = cv2.resize(face2, (right1-left1, bottom1-top1))
                    
                    # Replace faces
                    img1_cv[top1:bottom1, left1:right1] = face2
                    img2_cv[top2:bottom2, left2:right2] = face1
                    
                    # Convert back to RGB for display
                    result1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2RGB)
                    result2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2RGB)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(result1, caption="Image 1 with swapped face")
                    with col2:
                        st.image(result2, caption="Image 2 with swapped face")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                
                st.code("""
                import face_recognition
                import cv2
                import numpy as np
                
                # Load images
                image1 = face_recognition.load_image_file("person1.jpg")
                image2 = face_recognition.load_image_file("person2.jpg")
                
                # Find face locations
                face_locations1 = face_recognition.face_locations(image1)
                face_locations2 = face_recognition.face_locations(image2)
                
                # Convert to OpenCV format
                image1_cv = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
                image2_cv = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
                
                # Swap faces
                top1, right1, bottom1, left1 = face_locations1[0]
                top2, right2, bottom2, left2 = face_locations2[0]
                
                face1 = image1_cv[top1:bottom1, left1:right1]
                face2 = image2_cv[top2:bottom2, left2:right2]
                
                # Resize and swap
                face1 = cv2.resize(face1, (right2-left2, bottom2-top2))
                face2 = cv2.resize(face2, (right1-left1, bottom1-top1))
                
                image1_cv[top1:bottom1, left1:right1] = face2
                image2_cv[top2:bottom2, left2:right2] = face1
                
                # Save results
                cv2.imwrite("swapped1.jpg", image1_cv)
                cv2.imwrite("swapped2.jpg", image2_cv)
                """)
    
    with tab5:
        st.header("📚 Data Structures")
        
        ds_type = st.selectbox("Select data structure", ["List vs Tuple", "Dictionary", "Set"])
        
        if ds_type == "List vs Tuple":
            st.markdown("""
            ## Technical Difference Between Tuple and List
            
            | Feature        | List                                      | Tuple                                     |
            |---------------|-------------------------------------------|-------------------------------------------|
            | Mutability    | Mutable (can be changed after creation)   | Immutable (cannot be changed after creation) |
            | Syntax        | Uses square brackets `[]`                 | Uses parentheses `()`                     |
            | Performance   | Slightly slower due to mutability         | Faster due to immutability                |
            | Memory Usage  | More memory required                      | Less memory required                      |
            | Use Cases     | For collections that need to be modified  | For fixed collections of items            |
            
            **Example:**
            ```python
            # List example
            my_list = [1, 2, 3]
            my_list[0] = 10  # Valid - lists are mutable
            
            # Tuple example
            my_tuple = (1, 2, 3)
            my_tuple[0] = 10  # Invalid - tuples are immutable
            ```
            """)
    
    with tab6:
        st.header("🎙️ Voice Control")
        
        if st.button("Start Voice Command"):
            command = voice_input()
            if command:
                st.write(f"Executing command: {command}")
                
                # Simple voice commands
                if "home" in command.lower():
                    st.session_state.current_page = "Home"
                elif "linux" in command.lower():
                    st.session_state.current_page = "Linux"
                elif "aws" in command.lower():
                    st.session_state.current_page = "AWS"
                elif "git" in command.lower():
                    st.session_state.current_page = "Git"
                elif "python" in command.lower():
                    st.session_state.current_page = "Python"
                elif "machine learning" in command.lower() or "ml" in command.lower():
                    st.session_state.current_page = "ML"
                else:
                    st.warning("Command not recognized")

# Gemini Chat Page
def gemini_chat_page():
    st.title("💬 Gemini AI Assistant")
    
    if not st.session_state.gemini_model:
        st.warning("Please initialize Gemini AI from the Credentials page first")
        return
    
    # Initialize chat history
    if "gemini_chat" not in st.session_state:
        st.session_state.gemini_chat = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.gemini_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Ask Gemini anything..."):
        # Add user message to chat history
        st.session_state.gemini_chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from Gemini
        with st.spinner("Thinking..."):
            response = st.session_state.gemini_model.generate_content(prompt)
        
        # Add assistant response to chat history
        st.session_state.gemini_chat.append({"role": "assistant", "content": response.text})
        with st.chat_message("assistant"):
            st.markdown(response.text)

# Main App
def main():
    # Sidebar navigation
    with st.sidebar:
        st.title("🔧 Navigation")
        st.selectbox(
            "Go to",
            ["Home", "Credentials", "Linux", "ML", "AWS", "Git", "Python", "Gemini Chat"],
            key="current_page"
        )
        
        st.markdown("---")
        st.markdown("### Voice Command")
        if st.button("🎤 Use Voice Command"):
            command = voice_input()
            if command:
                process_voice_command(command)
        
        st.markdown("---")
        st.markdown("### System Status")
        if st.session_state.ssh_client:
            st.success("SSH: Connected")
        else:
            st.warning("SSH: Disconnected")
        
        if st.session_state.gemini_model:
            st.success("Gemini: Ready")
        else:
            st.warning("Gemini: Not initialized")
        
        if st.session_state.git_token:
            st.success("Git: Configured")
        else:
            st.warning("Git: Not configured")
        
        if st.session_state.aws_creds["access_key"]:
            st.success("AWS: Configured")
        else:
            st.warning("AWS: Not configured")
    
    # Page routing
    if st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.current_page == "Credentials":
        credentials_page()
    elif st.session_state.current_page == "Linux":
        linux_page()
    elif st.session_state.current_page == "ML":
        ml_page()
    elif st.session_state.current_page == "AWS":
        aws_page()
    elif st.session_state.current_page == "Git":
        git_page()
    elif st.session_state.current_page == "Python":
        python_page()
    elif st.session_state.current_page == "Gemini Chat":
        gemini_chat_page()

if __name__ == "__main__":
    main()