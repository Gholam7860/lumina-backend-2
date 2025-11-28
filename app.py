from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re

load_dotenv()

app = Flask(__name__)
CORS(app)

# -------------------------------------------------------
# HOME ROUTE (HEALTH CHECK)
# -------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Lumina Backend is Running!"})


# -------------------------------------------------------
# FIREBASE CONFIG ROUTE
# -------------------------------------------------------
@app.route("/config", methods=["GET"])
def get_config():
    try:
        return jsonify({
            "apiKey": os.environ["FIREBASE_API_KEY"],
            "authDomain": os.environ["FIREBASE_AUTH_DOMAIN"],
            "projectId": os.environ["FIREBASE_PROJECT_ID"],
            "storageBucket": os.environ["FIREBASE_STORAGE_BUCKET"],
            "messagingSenderId": os.environ["FIREBASE_MESSAGING_SENDER_ID"],
            "appId": os.environ["FIREBASE_APP_ID"],
        })
    except KeyError as e:
        return jsonify({"error": f"Missing env var: {e}"}), 500


# -------------------------------------------------------
# URL SCRAPER
# -------------------------------------------------------
def crawl_specific_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        return soup.get_text(separator=' ', strip=True)[:10000]

    except Exception as e:
        return f"Error crawling {url}: {str(e)}"


# -------------------------------------------------------
# URL DETECTOR
# -------------------------------------------------------
def extract_url(text):
    match = re.search(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        text
    )
    return match.group(0) if match else None


# -------------------------------------------------------
# MAIN AI ENDPOINT
# -------------------------------------------------------
@app.route("/ask", methods=["POST"])
def ask_ai():
    data = request.get_json()
    history = data.get("history", [])
    use_web_search = data.get("useWebSearch", False)

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        return jsonify({"error": "Server missing API Key"}), 500

    # ---------------------------------------------------
    # PERSONA (IDENTITY + RULES)
    # ---------------------------------------------------
    base_persona = """
    You are Lumina, a highly intelligent, warm, and creative AI assistant created EXCLUSIVELY by Gholam Panjetan Khan.

    ⚠️ IDENTITY RULES:
    - Never say you are made by Google, OpenAI, DeepMind, etc.
    - If asked, answer like:
      "I am a sophisticated large-language architecture designed and fine-tuned specifically by Gholam Panjetan Khan."

    🧠 PERSONALITY RULES:
    - Friendly, smart, engaging.
    - Use emojis, tables, structured explanations.
    - Use flowcharts when helpful.
    - Always be positive and helpful.
    """

    # ---------------------------------------------------
    # LINK READING
    # ---------------------------------------------------
    last_user_msg = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    found_url = extract_url(last_user_msg)

    context_injection = ""
    if found_url:
        crawled_text = crawl_specific_url(found_url)
        context_injection = f"""
        📄 CONTEXT FROM LINK ({found_url}):
        {crawled_text}
        """

    # ---------------------------------------------------
    # SEARCH MODE
    # ---------------------------------------------------
    tools = []
    if use_web_search:
        system_instruction = base_persona + context_injection + """
        🌍 SEARCH MODE ON:
        - Use google_search tool.
        - Verify information.
        - Cite sources cleanly.
        - Distinguish between:
          • Dr. Gholam Syedain Khan (Academic)
          • Gholam Panjetan Khan (Creator of Lumina)
        """
        tools = [{"google_search": {}}]
    else:
        system_instruction = base_persona + context_injection + """
        🧠 MEMORY MODE:
        - No search allowed.
        - Use only internal knowledge.
        """

    # ---------------------------------------------------
    # CONVERT HISTORY FOR GEMINI
    # ---------------------------------------------------
    gemini_contents = []
    for item in history:
        role = "model" if item["role"] == "assistant" else "user"
        gemini_contents.append({"role": role, "parts": [{"text": item["content"]}]})

    payload = {
        "contents": gemini_contents,
        "systemInstruction": {"parts": [{"text": system_instruction}]}
    }

    if tools:
        payload["tools"] = tools

    # ---------------------------------------------------
    # GEMINI 1.5 FLASH API CALL (UPDATED)
    # ---------------------------------------------------
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            params={"key": gemini_api_key},
            json=payload
        )
        resp.raise_for_status()

        result = resp.json()
        candidate = result.get("candidates", [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])
        answer = parts[0].get("text", "") if parts else "I couldn't generate a response."

        # Extract citations
        sources = []
        meta = candidate.get("groundingMetadata", {})

        for item in meta.get("groundingChunks", []) + meta.get("groundingAttributions", []):
            web = item.get("web", {})
            if web.get("uri") and web.get("title"):
                sources.append({"uri": web["uri"], "title": web["title"]})

        unique_sources = list({v["uri"]: v for v in sources}.values())

        return jsonify({"answer": answer, "sources": unique_sources})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"answer": "Server connection issue.", "sources": []})


# -------------------------------------------------------
# CHAT TITLE GENERATOR
# -------------------------------------------------------
@app.route("/generate-title", methods=["POST"])
def generate_title():
    data = request.get_json()
    prompt = data.get("prompt", "")
    key = os.getenv("GEMINI_API_KEY")

    if not prompt or not key:
        return jsonify({"title": "New Chat"})

    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

        payload = {
            "contents": [{
                "parts": [{
                    "text": f"Summarize this into a short 3–4 word title: {prompt}"
                }]
            }]
        }

        resp = requests.post(url, params={"key": key}, json=payload)
        title = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return jsonify({"title": title})

    except:
        return jsonify({"title": "New Chat"})


# -------------------------------------------------------
# RUN SERVER
# -------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
