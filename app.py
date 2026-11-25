import json
from datetime import datetime

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

import rag

app = Flask(__name__)


def read_question():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    hist = [{"role": "assistant" if h.get("role") == "assistant" else "user",
             "content": str(h.get("content", ""))[:600]}
            for h in (data.get("history") or [])[-6:] if isinstance(h, dict)]
    return q, hist


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/ask")
def ask():
    q, hist = read_question()
    if not q:
        return jsonify(error="Please type a question."), 400
    if len(q) > 500:
        return jsonify(error="Question is too long (max 500 characters)."), 400
    try:
        return jsonify(rag.answer(q, hist))
    except Exception as e:
        return jsonify(error=f"Service error: {e}"), 500


@app.post("/ask_stream")
def ask_stream():
    q, hist = read_question()
    if not q or len(q) > 500:
        return jsonify(error="Please type a question (max 500 characters)."), 400

    def gen():
        try:
            for ev, data in rag.stream_answer(q, hist):
                yield "data: " + json.dumps({"event": ev, "data": data}) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"event": "error", "data": f"Service error: {e}"}) + "\n\n"

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/feedback")
def feedback():
    d = request.get_json(silent=True) or {}
    row = {"time": datetime.now().isoformat(timespec="seconds"), "vote": d.get("vote"),
           "question": str(d.get("question", ""))[:500], "answer": str(d.get("answer", ""))[:1500]}
    with open("feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(debug=True)
