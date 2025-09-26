from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

app = Flask(__name__)
# Enable Cross-Origin Resource Sharing to allow the frontend to connect
CORS(app) 

# --- Configuration Endpoint ---
@app.route("/config", methods=["GET"])
def get_config():
    """Provides the necessary Firebase configuration to the frontend."""
    try:
        firebase_config = {
            "apiKey": os.environ["FIREBASE_API_KEY"],
            "authDomain": os.environ["FIREBASE_AUTH_DOMAIN"],
            "projectId": os.environ["FIREBASE_PROJECT_ID"],
            "storageBucket": os.environ["FIREBASE_STORAGE_BUCKET"],
            "messagingSenderId": os.environ["FIREBASE_MESSAGING_SENDER_ID"],
            "appId": os.environ["FIREBASE_APP_ID"],
        }
        return jsonify(firebase_config)
    except KeyError as e:
        # This error occurs if a required environment variable is missing
        return jsonify({"error": f"Missing environment variable: {e}"}), 500

# --- Main Chat Endpoint ---
@app.route("/ask", methods=["POST"])
def ask_ai():
    """Handles chat requests by forwarding them to the Gemini API."""
    data = request.get_json()
    conversation_history = data.get("history")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not conversation_history:
        return jsonify({"error": "No conversation history provided"}), 400
    if not gemini_api_key:
        return jsonify({"error": "GEMINI_API_KEY is not configured on the server"}), 500

    # API endpoint for the Gemini model
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": gemini_api_key}
    
    # Convert frontend role names ('user', 'assistant') to what the Gemini API expects ('user', 'model')
    gemini_contents = [
        {
            "role": "model" if item["role"] == "assistant" else "user",
            "parts": [{"text": item["content"]}]
        } 
        for item in conversation_history if item.get("content")
    ]

    # Construct the payload for the Gemini API call
    payload = {
        "contents": gemini_contents,
        "systemInstruction": {
            "parts": [{"text": "You are Lumina, an AI. Never mention Google or Gemini."}]
        },
        "tools": [{"google_search": {}}] # Enable Google Search for grounded responses
    }

    try:
        response = requests.post(url, headers=headers, params=params, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        result = response.json()
        
        # Safely extract the response text and grounding sources
        candidate = result.get("candidates", [{}])[0]
        content_part = candidate.get("content", {}).get("parts", [{}])[0]
        answer = content_part.get("text", "Sorry, an error occurred while processing the response.")
        
        grounding_metadata = candidate.get("groundingMetadata", {})
        attributions = grounding_metadata.get("groundingAttributions", [])
        
        sources = [
            {"uri": attr.get("web", {}).get("uri"), "title": attr.get("web", {}).get("title")}
            for attr in attributions if attr.get("web", {}).get("uri")
        ]
        
        # Ensure sources are unique
        unique_sources = list({v['uri']: v for v in sources}.values())

        return jsonify({"answer": answer, "sources": unique_sources})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to connect to the Gemini API", "details": str(e)}), 500
    except (KeyError, IndexError) as e:
        return jsonify({"error": "Could not parse the response from the Gemini API", "details": str(e)}), 500


# --- Title Generation Endpoint ---
@app.route("/generate-title", methods=["POST"])
def generate_title():
    """Generates a short title for a conversation based on an initial user prompt."""
    data = request.get_json()
    user_prompt = data.get("prompt")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not user_prompt:
        return jsonify({"error": "No prompt was provided"}), 400
    if not gemini_api_key:
        return jsonify({"error": "GEMINI_API_KEY is not configured on the server"}), 500
        
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": gemini_api_key}
    
    # A specific prompt to instruct the model on how to generate the title
    title_generation_prompt = f'Based on the following user query, create a short, descriptive title (3-5 words) for a chat conversation. Return only the title text, with no extra formatting or quotation marks. Query: "{user_prompt}"'
    payload = {"contents": [{"role": "user", "parts": [{"text": title_generation_prompt}]}]}
    
    try:
        response = requests.post(url, headers=headers, params=params, json=payload)
        response.raise_for_status()
        
        result = response.json()
        title = result["candidates"][0]["content"]["parts"][0]["text"].strip().replace('"', '')
        return jsonify({"title": title})
        
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to connect to the Gemini API for title generation", "details": str(e)}), 500
    except (KeyError, IndexError):
        # Provide a fallback title if parsing fails
        return jsonify({"title": "New Chat"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
