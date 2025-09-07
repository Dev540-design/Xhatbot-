from flask import Flask, request, render_template_string
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import requests
from bs4 import BeautifulSoup
import time
import ratelimit

app = Flask(__name__)

# Load transformer model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
last_request_time = [time.time()]  # Use list to allow updates inside function

# Simple HTML template
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Transformer Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .chatbox { width: 500px; margin: auto; }
        textarea, input { width: 100%; }
        .response { background: #eef; padding: 10px; margin-top: 20px; }
    </style>
</head>
<body>
<div class="chatbox">
    <h2>Transformer Chatbot</h2>
    <form method="POST">
        <textarea name="message" rows="3" placeholder="Type a message, or 'scrape <url>'"></textarea><br>
        <input type="submit" value="Send">
    </form>
    {% if response %}
    <div class="response"><b>Bot:</b><br>{{ response }}</div>
    {% endif %}
</div>
</body>
</html>
"""

@ratelimit.limits(calls=10, period=60)
def rate_limited_get_url_content(url):
    try:
        # Enforce 1 second minimum between requests
        current_time = time.time()
        if current_time - last_request_time[0] < 1:
            time.sleep(1 - (current_time - last_request_time[0]))
        last_request_time[0] = time.time()

        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Basic check for robots.txt (not a full parser)
        # In production, use proper robots.txt checking!
        if 'Disallow' in response.text:
            return "Scraping not permitted by website."

        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text
    except requests.exceptions.RequestException as e:
        return f"Error accessing URL: {e}"
    except Exception as e:
        return f"An unexpected error occurred during scraping: {e}"

def generate_response(user_input):
    user_input_lower = user_input.lower()
    if user_input_lower.startswith("scrape "):
        url = user_input[len("scrape "):].strip()
        if url:
            scraped_text = rate_limited_get_url_content(url)
            if scraped_text.startswith("Error") or scraped_text.startswith("Scraping not permitted"):
                return scraped_text
            else:
                preview = scraped_text[:500] + ('...' if len(scraped_text) > 500 else '')
                return f"Successfully scraped content from {url}:<br><pre>{preview}</pre>"
        else:
            return "Please provide a URL to scrape."
    # Otherwise, use transformer model
    inputs = tokenizer.encode("question: " + user_input, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
    model_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if not model_response or model_response.strip() == "":
        return "I'm not sure how to respond to that. Could you rephrase?"
    return model_response

@app.route("/", methods=["GET", "POST"])
def home():
    response = None
    if request.method == "POST":
        msg = request.form.get("message", "")
        response = generate_response(msg)
    return render_template_string(HTML, response=response)

if __name__ == "__main__":
    app.run(debug=True)
