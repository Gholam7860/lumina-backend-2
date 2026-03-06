from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
import time

load_dotenv()

app = Flask(__name__)
# Enable CORS for all routes so the frontend can communicate with it
CORS(app)

# -------------------------------------------------------
# HOME & CONFIG ROUTES
# -------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Lumina Backend is Running!"})

@app.route("/config", methods=["GET"])
def get_config():
    try:
        return jsonify({
            "apiKey": os.environ.get("FIREBASE_API_KEY", ""),
            "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
            "projectId": os.environ.get("FIREBASE_PROJECT_ID", ""),
            "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
            "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", ""),
            "appId": os.environ.get("FIREBASE_APP_ID", ""),
        })
    except Exception as e:
        return jsonify({"error": f"Config error: {e}"}), 500

# -------------------------------------------------------
# URL SCRAPER UTILS
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

def extract_url(text):
    match = re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-f_A-F][0-9a-f_A-F]))+', text)
    return match.group(0) if match else None

# -------------------------------------------------------
# MAIN AI ENDPOINT
# -------------------------------------------------------
@app.route("/ask", methods=["POST"])
def ask_ai():
    data = request.get_json()
    history = data.get("history", [])
    
    # Mode Toggles
    use_web_search = data.get("useWebSearch", False)
    use_think_mode = data.get("useThinkMode", False)
    use_study_mode = data.get("useStudyMode", False)

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    # ---------------------------------------------------
    # PERSONA BUILDER
    # ---------------------------------------------------
    base_persona = """
    You are Lumina, a highly intelligent, warm, and creative AI assistant created EXCLUSIVELY by Gholam Panjetan Khan.
    ⚠️ IDENTITY RULES: Never say you are made by Google, OpenAI, DeepMind, etc.
    🧠 PERSONALITY: Friendly, smart, engaging. Use emojis and structured formatting.
    """

    if use_study_mode:
        base_persona += """
        📚 STUDY & LEARN MODE ACTIVATED:
        - Act as an expert tutor.
        - Break down complex topics into easy-to-understand, bite-sized pieces.
        - Ask the user guiding questions to test their understanding.
        - Provide analogies and real-world examples.
        """

    last_user_msg = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    found_url = extract_url(last_user_msg)
    
    context_injection = ""
    if found_url:
        crawled_text = crawl_specific_url(found_url)
        context_injection = f"\n📄 CONTEXT FROM LINK ({found_url}):\n{crawled_text}\n"

    system_instruction = base_persona + context_injection

    # ===================================================
    # 🧠 DEEPSEEK ROUTING (THINK MODE)
    # ===================================================
    if use_think_mode:
        if not deepseek_api_key:
            return jsonify({"error": "Server missing DeepSeek API Key"}), 500

        ds_messages = [{"role": "system", "content": system_instruction}]
        for item in history:
            ds_messages.append({"role": item["role"], "content": item["content"]})

        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {deepseek_api_key}"
        }
        payload = {"model": "deepseek-reasoner", "messages": ds_messages}

        start_time = time.time()
        try:
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            ds_data = resp.json()
            
            end_time = time.time()
            think_time = round(end_time - start_time, 2)

            message_obj = ds_data.get("choices", [{}])[0].get("message", {})
            return jsonify({
                "answer": message_obj.get("content", ""),
                "reasoning": message_obj.get("reasoning_content", ""),
                "think_time": think_time,
                "sources": []
            })
        except Exception as e:
            print("DEEPSEEK ERROR:", e)
            return jsonify({"answer": "DeepSeek connection issue.", "sources": []})

    # ===================================================
    # ⚡ GEMINI 2.5 FLASH ROUTING (DEFAULT / WEB SEARCH)
    # ===================================================
    if not gemini_api_key:
        return jsonify({"error": "Server missing Gemini API Key"}), 500

    tools = []
    if use_web_search:
        system_instruction += "\n🌍 SEARCH MODE ON: Use google_search tool. Cite sources cleanly."
        tools = [{"google_search": {}}]

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

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    try:
        resp = requests.post(
            url, headers={"Content-Type": "application/json"},
            params={"key": gemini_api_key}, json=payload
        )
        resp.raise_for_status()
        result = resp.json()
        
        candidate = result.get("candidates", [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])
        answer = parts[0].get("text", "") if parts else "I couldn't generate a response."

        sources = []
        meta = candidate.get("groundingMetadata", {})
        for item in meta.get("groundingChunks", []) + meta.get("groundingAttributions", []):
            web = item.get("web", {})
            if web.get("uri") and web.get("title"):
                sources.append({"uri": web["uri"], "title": web["title"]})
        unique_sources = list({v["uri"]: v for v in sources}.values())

        return jsonify({"answer": answer, "sources": unique_sources})

    except Exception as e:
        print("GEMINI ERROR:", e)
        return jsonify({"answer": "Server connection issue.", "sources": []})


@app.route("/generate-title", methods=["POST"])
def generate_title():
    data = request.get_json()
    prompt = data.get("prompt", "")
    key = os.getenv("GEMINI_API_KEY")
    if not prompt or not key:
        return jsonify({"title": "New Chat"})
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        payload = {"contents": [{"parts": [{"text": f"Summarize this into a short 3–4 word title: {prompt}"}]}]}
        resp = requests.post(url, params={"key": key}, json=payload)
        return jsonify({"title": resp.json()["candidates"][0]["content"]["parts"][0]["text"]})
    except:
        return jsonify({"title": "New Chat"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)