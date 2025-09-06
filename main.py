import os
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

BOT_TOKEN = "8304180767:AAEGUvEv9TnFNoqZ0R90pASPB4p8dIjAYTk"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"


@app.route("/", methods=["GET"])
def home():
    return "Bot is running! 🚀"


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json

    if "message" not in data:  # safety check
        return jsonify({"ok": False}), 400

    chat_id = data["message"]["chat"]["id"]
    text = data["message"].get("text", "")

    reply = f"You said: {text}"
    requests.post(f"{TELEGRAM_API_URL}/sendMessage", json={
        "chat_id": chat_id,
        "text": reply
    })

    return jsonify({"ok": True})


@app.route("/send", methods=["POST"])
def send_message():
    data = request.json
    chat_id = data.get("chat_id")
    text = data.get("text")

    if not chat_id or not text:
        return jsonify({"error": "chat_id and text are required"}), 400

    response = requests.post(f"{TELEGRAM_API_URL}/sendMessage", json={
        "chat_id": chat_id,
        "text": text
    })

    return jsonify(response.json())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
