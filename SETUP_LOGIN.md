# Setting Up Google Login (for Private Chat History)

Your app now requires logging in with Google before use -- this is what
lets each person see only THEIR OWN chat history, instead of one shared
history for everyone. This needs a free Google Cloud project to issue
the login credentials. Takes about 10 minutes, one-time setup.

## Step 1: Create a Google Cloud project (free)
1. Go to https://console.cloud.google.com
2. Sign in with any Google account
3. Click the project dropdown (top left) → "New Project"
4. Name it anything (e.g. "ai-assistant-login") → Create

## Step 2: Configure the OAuth consent screen
1. In the left menu, go to "APIs & Services" → "OAuth consent screen"
2. Choose "External" user type → Create
3. Fill in: App name (e.g. "My AI Assistant"), your email for both
   "User support email" and "Developer contact"
4. Click through the remaining steps (Scopes, Test users) using the
   defaults → Save and Continue → Back to Dashboard

## Step 3: Create OAuth credentials
1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth client ID"
3. Application type: "Web application"
4. Name: anything (e.g. "Streamlit AI Assistant")
5. Under "Authorized redirect URIs", click "Add URI" and enter EXACTLY:
   ```
   https://YOUR-APP-NAME.streamlit.app/oauth2callback
   ```
   Replace `YOUR-APP-NAME` with your actual Streamlit app subdomain.
   This must match exactly, including `https://` and the path.
6. Click "Create"
7. A popup shows your **Client ID** and **Client Secret** -- copy both
   somewhere safe, you'll need them in Step 5.

## Step 4: Generate a cookie secret
This is just a random string Streamlit uses to securely sign login
cookies -- any strong random string works. Generate one by running
this in any Python environment (or your terminal):
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```
Copy the output.

## Step 5: Add everything to Streamlit secrets
1. On your Streamlit Cloud app dashboard, go to Settings → Secrets
2. Add this block (using YOUR actual values from Steps 3 and 4):
   ```toml
   GROQ_API_KEY = "your-existing-groq-key-here"

   [auth]
   redirect_uri = "https://YOUR-APP-NAME.streamlit.app/oauth2callback"
   cookie_secret = "the-random-string-from-step-4"
   client_id = "your-client-id-from-step-3"
   client_secret = "your-client-secret-from-step-3"
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
   ```
3. Save -- the app will restart automatically

## Step 6: Test it
Open your app URL. You should now see a "Log in with Google" button
before anything else loads. Log in, and you'll see the chat interface
with a "History" section in the sidebar -- empty at first, since this
is your first login.

## Important notes
- **Your app is in "Testing" mode by default** (from Step 2), which
  means only email addresses you explicitly add as "Test users" in the
  OAuth consent screen can log in. To let anyone log in, you'd need to
  "Publish" the app in Google Cloud Console, which requires a brief
  Google verification review for apps requesting certain scopes (basic
  login doesn't usually need this, but double-check under "Publishing
  status" in the consent screen settings).
- **History storage is not fully permanent.** Conversations are saved
  to a local database file on Streamlit Cloud's server. This survives
  normal use, but Streamlit Cloud's free tier can reset the filesystem
  on certain events (e.g., redeploys after a `git push`, or occasional
  platform maintenance) -- so treat this as "history that sticks around
  for normal day-to-day use," not "permanent forever" storage. For
  guaranteed permanent history, you'd eventually want an external
  hosted database -- a bigger, separate upgrade if you need it later.
- **Local testing**: if you want to test login locally before pushing,
  you'd need a second OAuth client with `redirect_uri` set to
  `http://localhost:8501/oauth2callback`, added to your local
  `.streamlit/secrets.toml`. Not required if you're only using the
  deployed version.
