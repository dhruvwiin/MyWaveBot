# Deployment Instructions for Render.com (Free Tier)

This application is ready to be deployed to the cloud. We recommend **Render.com** because it has a generous free tier for web services and is very easy to set up.

## Prerequisites
1.  **GitHub Account**: You need to push this code to a GitHub repository.
2.  **Render Account**: Sign up at [render.com](https://render.com).

## Step 1: Push Code to GitHub
If you haven't already, initialize a git repository and push this code to GitHub.

```bash
git init
git add .
git commit -m "Initial commit"
# Create a new repo on GitHub, then:
git remote add origin <your-github-repo-url>
git push -u origin main
```

## Step 2: Create a Web Service on Render
1.  Log in to your Render dashboard.
2.  Click **New +** and select **Web Service**.
3.  Connect your GitHub account and select the repository you just pushed.
4.  Configure the service:
    *   **Name**: `tulane-chatbot` (or any name you like)
    *   **Region**: Choose the one closest to you (e.g., US East).
    *   **Branch**: `main`
    *   **Root Directory**: (Leave blank)
    *   **Runtime**: `Python 3`
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `uvicorn server.app:app --host 0.0.0.0 --port $PORT`
    *   **Instance Type**: Select **Free**.

## Step 3: Configure Environment Variables
Scroll down to the **Environment Variables** section and add the following keys (copy them from your local `.env` file):

| Key | Value |
| --- | --- |
| `PERPLEXITY_API_KEY` | `pplx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` |
| `ADMIN_API_TOKEN` | `your_secure_admin_token_here` |
| `HASH_SALT` | `your_random_salt_value_here` |
| `UNIVERSITY_NAME` | `Tulane University` |
| `SEARCH_DOMAINS` | `tulane.edu,registrar.tulane.edu,housing.tulane.edu,admission.tulane.edu,financialaid.tulane.edu,catalog.tulane.edu,library.tulane.edu,campushealth.tulane.edu,campusservices.tulane.edu,it.tulane.edu,careers.tulane.edu,studentaffairs.tulane.edu` |
| `STORE_MESSAGE_TEXT` | `True` |

## Step 4: Deploy
Click **Create Web Service**. Render will start building your application. It usually takes 2-3 minutes.

Once finished, you will get a URL like `https://tulane-chatbot.onrender.com`. You can share this URL with anyone!

## Note on Database
On the free tier, the SQLite database (`analytics.db`) is **ephemeral**. This means if the server restarts (which happens occasionally on free tiers), the chat history and analytics will be reset. For a permanent database, you would need to upgrade to a paid plan or connect an external database (like Render's PostgreSQL), but for a demo, this is fine.
