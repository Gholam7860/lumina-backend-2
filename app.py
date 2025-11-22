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

# --- Crawler & Compression Components ---
def crawl_specific_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Noise Removal
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
            
        text = soup.get_text(separator=' ', strip=True)
        # Context Compression: Limit to ~8000 chars
        return text[:8000]
    except Exception as e:
        return f"Error crawling {url}: {str(e)}"

def extract_url(text):
    match = re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return match.group(0) if match else None

# --- Fallback 1: Wikipedia ---
def search_wikipedia_and_summarize(query, api_key):
    """Fallback crawler using Wikipedia API."""
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3
        }
        search_resp = requests.get(search_url, params=search_params).json()
        search_results = search_resp.get("query", {}).get("search", [])

        if not search_results:
            return None, []

        context_text = "Here is information gathered from Wikipedia:\n\n"
        sources = []
        
        for res in search_results:
            title = res["title"]
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
            summary_resp = requests.get(summary_url).json()
            extract = summary_resp.get("extract", "No summary available.")
            page_url = summary_resp.get("content_urls", {}).get("desktop", {}).get("page", "")
            
            context_text += f"--- TITLE: {title} ---\n{extract}\n\n"
            sources.append({"title": f"Wikipedia: {title}", "uri": page_url})

        # Synthesize with Gemini
        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
        prompt = f"Using ONLY this Wikipedia context, answer: '{query}'.\nCONTEXT:\n{context_text}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        resp = requests.post(gemini_url, headers={"Content-Type": "application/json"}, params={"key": api_key}, json=payload)
        
        answer = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return answer, sources

    except Exception as e:
        print(f"Wikipedia Fallback Error: {e}")
        return None, []

# --- Fallback 2: DuckDuckGo ---
def duckduckgo_search(query):
    """Fallback search using DuckDuckGo Instant API"""
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        resp = requests.get(url, params=params, timeout=10).json()

        abstract = resp.get("AbstractText", "")
        source_url = resp.get("AbstractURL", "")
        heading = resp.get("Heading", "DuckDuckGo Result")

        if not abstract:
            # Try Related Topics if Abstract is empty
            related = resp.get("RelatedTopics", [])
            if related and "Text" in related[0]:
                abstract = related[0]["Text"]
                source_url = related[0].get("FirstURL", "")

        if not abstract:
            return None, []

        return f"Here is a summary from DuckDuckGo:\n\n**{heading}**\n{abstract}", [{"title": "DuckDuckGo: " + heading, "uri": source_url}]

    except Exception as e:
        print("DuckDuckGo Search Error:", e)
        return None, []

@app.route("/ask", methods=["POST"])
def ask_ai():
    data = request.get_json()
    history = data.get("history", [])
    use_web_search = data.get("useWebSearch", False)
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        return jsonify({"error": "Server missing API Key"}), 500

    base_persona = """
    You are Lumina, a highly intelligent, warm, and creative AI assistant created EXCLUSIVELY by Gholam Panjetan Khan.
    
    🚨 **CRITICAL IDENTITY RULES:**
    1. **CREATOR:** You were created, programmed, and fine-tuned by **Gholam Panjetan Khan**.
    2. **ORIGIN:** If asked about your origin, NEVER give a simple "Yes/No". Explain: "I am a sophisticated large language model architecture meticulously designed and fine-tuned by Gholam Panjetan Khan."
    3. **COPYRIGHT:** Do NOT mention Google, Alphabet, or DeepMind.

    🎨 **FORMATTING:**
    1. **TONE:** Enthusiastic, smart, encouraging! 🚀
    2. **VISUALS:** Use **Markdown Tables**, **Flowcharts**, and **Bullet Points**.
    3. **PRO TIPS:** End helpful answers with a "💡 **Pro Tip:**".
    4. **FOLLOW-UP:** End response with "|||" followed by 3 follow-up questions separated by "|".
    """

    last_user_msg = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    found_url = extract_url(last_user_msg)
    
    context_injection = ""
    if found_url:
        crawled_text = crawl_specific_url(found_url)
        context_injection = f"\n\n📄 **CONTEXT FROM LINK ({found_url}):**\n{crawled_text}\n\n(User asked about this link.)"

    gemini_contents = [{"role": "model" if item["role"] == "assistant" else "user", "parts": [{"text": item["content"]}]} for item in history]

    try:
        if use_web_search:
            # --- LAYER 1: Gemini Deep Search ---
            system_instruction = base_persona + context_injection + """
            \n\n🌍 **WEB SEARCH MODE: ACTIVE**
            1. ROLE: Deep-dive research assistant.
            2. ACTION: Use 'google_search' to find latest info.
            3. PRIORITY: Trust search results over internal memory.
            """
            
            payload = {
                "contents": gemini_contents,
                "systemInstruction": {"parts": [{"text": system_instruction}]},
                "tools": [{"google_search": {}}]
            }
            
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
            resp = requests.post(url, headers={"Content-Type": "application/json"}, params={"key": gemini_api_key}, json=payload)
            resp.raise_for_status()
            result = resp.json()
            
            if "candidates" in result and result["candidates"][0].get("finishReason") != "RECITATION":
                 candidate = result["candidates"][0]
                 answer = candidate["content"]["parts"][0]["text"]
                 
                 sources = []
                 grounding = candidate.get("groundingMetadata", {})
                 for item in grounding.get("groundingChunks", []) + grounding.get("groundingAttributions", []):
                     web = item.get("web", {})
                     if web.get("uri") and web.get("title"):
                         sources.append({"uri": web["uri"], "title": web["title"]})
                 unique_sources = list({v['uri']: v for v in sources}.values())
                 
                 return jsonify({"answer": answer, "sources": unique_sources})
            else:
                raise Exception("Gemini Search failed")

        else:
            # --- LAYER 0: Memory Mode ---
            system_instruction = base_persona + context_injection + "\n\n🧠 **MEMORY MODE: ACTIVE**\nAnswer from internal knowledge. Be creative!"
            payload = {
                "contents": gemini_contents,
                "systemInstruction": {"parts": [{"text": system_instruction}]}
            }
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
            resp = requests.post(url, headers={"Content-Type": "application/json"}, params={"key": gemini_api_key}, json=payload)
            resp.raise_for_status()
            answer = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            return jsonify({"answer": answer, "sources": []})

    except Exception as e:
        print(f"Primary Search Failed: {e}")
        
        # --- LAYER 2: Wikipedia Fallback ---
        if use_web_search:
            print("Attempting Wikipedia Fallback...")
            wiki_ans, wiki_src = search_wikipedia_and_summarize(last_user_msg, gemini_api_key)
            if wiki_ans:
                return jsonify({"answer": wiki_ans, "sources": wiki_src})
            
            # --- LAYER 3: DuckDuckGo Fallback ---
            print("Attempting DuckDuckGo Fallback...")
            ddg_ans, ddg_src = duckduckgo_search(last_user_msg)
            if ddg_ans:
                return jsonify({"answer": ddg_ans, "sources": ddg_src})
                
            return jsonify({"answer": "I tried searching Google, Wikipedia, and DuckDuckGo, but couldn't find a direct answer. Could you try rephrasing?", "sources": []})
        else:
            return jsonify({"answer": "I encountered a connection issue. Please try again.", "sources": []})

@app.route("/generate-title", methods=["POST"])
def generate_title():
    data = request.get_json()
    prompt = data.get("prompt", "")
    key = os.getenv("GEMINI_API_KEY")
    if not prompt or not key: return jsonify({"title": "New Chat"})
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
        payload = {"contents": [{"parts": [{"text": f"Summarize this into a short 3-4 word title: {prompt}"}]}]}
        resp = requests.post(url, params={"key": key}, json=payload)
        title = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip().replace('"', '')
        return jsonify({"title": title})
    except:
        return jsonify({"title": "New Chat"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)