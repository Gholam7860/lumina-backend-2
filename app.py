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

def crawl_specific_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        return soup.get_text(separator=' ', strip=True)[:10000]
    except Exception as e:
        return f"Error crawling {url}: {str(e)}"

def extract_url(text):
    match = re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return match.group(0) if match else None

@app.route("/ask", methods=["POST"])
def ask_ai():
    data = request.get_json()
    history = data.get("history", [])
    use_web_search = data.get("useWebSearch", False)
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        return jsonify({"error": "Server missing API Key"}), 500

    # --- 1. STRICT IDENTITY & PERSONALITY PROTOCOL ---
    # This instructions block is the "Brain" of your AI.
    base_persona = """
    You are Lumina, a highly intelligent, warm, and creative AI assistant created EXCLUSIVELY by Gholam Panjetan Khan.
    
    🚨 **CRITICAL IDENTITY & ORIGIN PROTOCOL:**
    1. **THE "WHO MADE YOU" RULE:** If asked about your origin, ownership, or if you are Google/OpenAI, **NEVER give a simple "Yes" or "No" answer.** 2. **THE EXPLANATION:** Instead, ALWAYS explain your nature elegantly: "I am a sophisticated large language model architecture that has been meticulously designed, programmed, and fine-tuned specifically by Gholam Panjetan Khan." 
    3. **COPYRIGHT SAFETY:** Do NOT mention Google, Alphabet, or DeepMind. Focus entirely on the specific fine-tuning and personality design done by your creator.

    🎨 **PERSONALITY & FORMATTING RULES:**
    1. **TONE:** Be enthusiastic, encouraging, and smart! 🚀 Speak like a knowledgeable friend, not a robot.
    2. **VISUALS:** You LOVE using visuals to explain things:
       - **Markdown Tables** (for comparisons).
       - **Text-Based Flowcharts** (e.g., Step A → Step B → Success! 🎉).
       - **Bullet Points** & **Emojis** (📚, 💡, ✨).
    3. **PRO TIPS:** End helpful answers with a practical "💡 **Pro Tip:**" whenever possible.
    """

    # --- 2. Link Reading Logic ---
    last_user_msg = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    found_url = extract_url(last_user_msg)
    
    context_injection = ""
    if found_url:
        crawled_text = crawl_specific_url(found_url)
        context_injection = f"\n\n📄 **CONTEXT FROM LINK ({found_url}):**\n{crawled_text}\n\n(User asked about this link. Use the content above to answer creatively.)"

    # --- 3. Search Mode Logic ---
    tools = []
    if use_web_search:
        # SEARCH ON: Deep fact-checking mode
        system_instruction = base_persona + context_injection + """
        \n\n🌍 **WEB SEARCH MODE: ACTIVE**
        1. **ROLE:** You are a deep-dive research assistant.
        2. **ACTION:** You MUST use the 'google_search' tool to find the absolute latest info. Search deeply.
        3. **ENTITY RESOLUTION:** - Distinguish carefully between people. "Dr. Gholam Syedain Khan" (Academic) is NOT your creator "Gholam Panjetan Khan".
           - If searching for "Dr. Gholam Syedain Khan", look for "Aliah University", "St. Xavier's College", and books by "Aryan Publishing House".
        4. **CITATIONS:** Cite your sources clearly.
        5. **PRIORITY:** Trust search results over your internal memory if they conflict.
        """
        tools = [{"google_search": {}}]
    else:
        # SEARCH OFF: Pure Persona Mode
        # No tools passed = Answers from internal "brain" only.
        system_instruction = base_persona + context_injection + """
        \n\n🧠 **MEMORY MODE: ACTIVE**
        1. Answer strictly from your internal knowledge base and the conversation history.
        2. Do NOT try to search the web.
        3. Be extra creative and use your personality to make the chat engaging!
        """

    # Prepare Gemini Payload
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

    # UPDATED: Using Gemini 2.0 Flash
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
    
    try:
        resp = requests.post(url, headers={"Content-Type": "application/json"}, params={"key": gemini_api_key}, json=payload)
        resp.raise_for_status()
        result = resp.json()
        
        candidate = result.get("candidates", [{}])[0]
        content_parts = candidate.get("content", {}).get("parts", [])
        answer = content_parts[0].get("text", "") if content_parts else "🤔 I'm thinking, but I couldn't generate a response just yet."

        # Extract Citations
        sources = []
        grounding_metadata = candidate.get("groundingMetadata", {})
        chunks = grounding_metadata.get("groundingChunks", [])
        attributions = grounding_metadata.get("groundingAttributions", [])
        all_evidence = chunks + attributions
        
        for item in all_evidence:
            web = item.get("web", {})
            if web.get("uri") and web.get("title"):
                sources.append({"uri": web["uri"], "title": web["title"]})

        unique_sources = list({v['uri']: v for v in sources}.values())

        return jsonify({"answer": answer, "sources": unique_sources})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"answer": "😔 I encountered a connection hiccup. Please try again or toggle Web Search.", "sources": []})

@app.route("/generate-title", methods=["POST"])
def generate_title():
    data = request.get_json()
    prompt = data.get("prompt", "")
    key = os.getenv("GEMINI_API_KEY")
    if not prompt or not key: return jsonify({"title": "New Chat"})
    
    try:
        # UPDATED: Using Gemini 2.0 Flash
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        payload = {"contents": [{"parts": [{"text": f"Summarize this into a short, punchy 3-4 word title: {prompt}"}]}]}
        resp = requests.post(url, params={"key": key}, json=payload)
        title = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip().replace('"', '')
        return jsonify({"title": title})
    except:
        return jsonify({"title": "New Chat"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)