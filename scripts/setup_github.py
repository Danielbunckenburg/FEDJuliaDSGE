import os
import requests
import subprocess
from dotenv import load_dotenv

def setup_github():
    # Load credentials
    if not os.path.exists(".env"):
        print("Error: .env file not found. Please create one from .env.template")
        return

    load_dotenv()
    username = os.getenv("GITHUB_USERNAME")
    token = os.getenv("GITHUB_TOKEN")
    repo_name = os.path.basename(os.getcwd())

    if not username or not token:
        print("Error: GITHUB_USERNAME or GITHUB_TOKEN not set in .env")
        return

    print(f"Targeting repository: {username}/{repo_name}")

    # 1. Create Repository via GitHub API
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "name": repo_name,
        "private": False # Change to True if you want a private repo
    }

    print("Creating repository on GitHub...")
    response = requests.post("https://api.github.com/user/repos", headers=headers, json=data)

    if response.status_code == 201:
        print("Successfully created repository on GitHub.")
    elif response.status_code == 422:
        print("Repository already exists on GitHub. Proceeding to push...")
    else:
        print(f"Failed to create repository: {response.status_code}")
        print(response.json())
        return

    # 2. Push to GitHub
    remote_url = f"https://{username}:{token}@github.com/{username}/{repo_name}.git"
    
    try:
        # Check if remote already exists
        remotes = subprocess.check_output(["git", "remote"], stderr=subprocess.STDOUT).decode()
        if "origin" in remotes:
            subprocess.run(["git", "remote", "set-url", "origin", remote_url])
        else:
            subprocess.run(["git", "remote", "add", "origin", remote_url])
        
        print("Pushing to GitHub...")
        subprocess.run(["git", "branch", "-M", "main"])
        subprocess.run(["git", "push", "-u", "origin", "main"])
        print("Done!")
    except Exception as e:
        print(f"Git execution failed: {e}")

if __name__ == "__main__":
    setup_github()
