import base64
import io
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_dotenv():
    env_path = os.path.join(ROOT_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()

HOST = os.environ.get("FINDER_HOST", "127.0.0.1")
PORT = int(os.environ.get("FINDER_PORT", "3000"))

CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")

PROVIDERS = {
    "gemini": {
        "name": "Google Gemini",
        "model": "gemini-3.5-flash-lite",
        "key": os.environ.get("GEMINI_API_KEY", ""),
    },
    "groq": {
        "name": "Groq",
        "model": "openai/gpt-oss-120b",
        "key": os.environ.get("GROQ_API_KEY", ""),
        "url": "https://api.groq.com/openai/v1/chat/completions",
    },
    "cloudflare": {
        "name": "Cloudflare AI",
        "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "key": CF_API_TOKEN,
    },
    "openrouter": {
        "name": "OpenRouter",
        "model": "openai/gpt-4o-mini",
        "key": os.environ.get("OPENROUTER_API_KEY", ""),
        "url": "https://openrouter.ai/api/v1/chat/completions",
    },
    "cohere": {
        "name": "Cohere",
        "model": "command-a-plus-05-2026",
        "key": COHERE_API_KEY,
        "url": "https://api.cohere.com/v2/chat",
    },
}

THINK_ORDER = ["gemini", "groq", "cohere", "cloudflare", "openrouter"]
AUTO_ORDER = ["gemini", "cohere", "groq", "cloudflare", "openrouter"]
FAILOVER_ORDER = ["gemini", "cohere", "groq", "cloudflare", "openrouter"]
IMAGE_MODELS = [
    "@cf/black-forest-labs/flux-1-schnell",
    "@cf/black-forest-labs/flux-2-klein-4b",
    "@cf/leonardo/lucid-origin",
]
PROMPT_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"


def clean_account_id(raw: str | None) -> str:
    val = re.sub(r"[^a-zA-Z0-9_-]", "", raw or "")
    return val or CF_ACCOUNT_ID


def resolve_token(auth_header: str | None, fallback: str) -> str:
    if auth_header and auth_header.startswith("Bearer "):
        candidate = auth_header[7:].strip()
        if candidate:
            return candidate
    return fallback


def translate_prompt_to_english(prompt: str, account_id: str, token: str) -> str:
    """Translate/refine non-English visual prompts into concise English for image generation."""
    stripped = re.sub(
        r"^(por favor\s+|pode\s+|consegue\s+|quero\s+|me\s+)?(crie|cria|criar|gera|gere|gerar|faz|faça|fazer|desenhe|desenha|desenhar|ilustre|ilustra|pinte|pinta|create|generate|draw|make)\s+(aí\s+|para mim\s+|pra mim\s+)?(uma\s+|um\s+|a\s+|an\s+)?(imagem|foto|fotografia|ilustração|arte|desenho|capa|logo|banner|avatar|wallpaper|papel de parede|retrato|picture|image|photo|illustration)?\s*(de|da|do|dos|das|of|for)?\s*",
        "",
        prompt.strip(),
        flags=re.IGNORECASE,
    ).strip() or prompt.strip()

    if all(ord(ch) < 128 for ch in stripped) and not re.search(r"\b(de|da|do|com|uma|um|para|em|no|na)\b", stripped, re.I):
        return stripped

    payload = json.dumps({
        "model": PROMPT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Convert the user's image request into a concise, vivid English image-generation prompt. Output ONLY the prompt text, nothing else.",
            },
            {"role": "user", "content": stripped},
        ],
        "max_tokens": 120,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
            return text or stripped
    except Exception:
        return stripped


def format_cf_error(status: int, raw_body: bytes) -> tuple[int, str]:
    detail = ""
    try:
        parsed = json.loads(raw_body.decode("utf-8", errors="replace"))
        errs = parsed.get("errors") or []
        if errs and isinstance(errs, list):
            detail = errs[0].get("message", "")
        elif isinstance(parsed.get("error"), dict):
            detail = parsed["error"].get("message", "")
    except Exception:
        detail = raw_body.decode("utf-8", errors="replace")[:200]

    lower = detail.lower()
    if status == 429 or any(k in lower for k in ("quota", "neuron", "rate limit", "daily limit", "exceeded")):
        return 429, "O limite diário gratuito de 10.000 Neurons ou de requisições por minuto da Cloudflare foi atingido. Tente novamente mais tarde (reset às 00:00 UTC)."
    if status in (401, 403):
        return status, "Token da Cloudflare recusado ou sem permissão para Workers AI."
    return status or 502, detail or "Não foi possível gerar a imagem na Cloudflare."


def open_chat_upstream(prov_id: str, model: str, key: str, messages: list, sys_instr: str, account_id: str, think: bool = False):
    """Open an SSE upstream response for a given provider."""
    if prov_id == "gemini":
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        body = {"contents": contents}
        if sys_instr:
            body["systemInstruction"] = {"parts": [{"text": sys_instr}]}
        if think:
            body["generationConfig"] = {"thinkingConfig": {"thinkingBudget": 8192}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    elif prov_id == "cohere":
        api_msgs = []
        if sys_instr:
            api_msgs.append({"role": "system", "content": sys_instr})
        for m in messages:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                api_msgs.append({"role": m["role"], "content": m["content"]})
        body = {"model": model, "messages": api_msgs, "stream": True}
        url = PROVIDERS["cohere"]["url"]
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    else:
        api_msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        if sys_instr:
            api_msgs.insert(0, {"role": "system", "content": sys_instr})
        body = {"model": model, "messages": api_msgs, "stream": True}
        if think and "gpt-oss" in model:
            body["reasoning_effort"] = "high"
        if prov_id == "cloudflare":
            url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        else:
            url = PROVIDERS[prov_id]["url"]
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
            if prov_id == "openrouter":
                headers["HTTP-Referer"] = "http://localhost:3000"
                headers["X-Title"] = "FINDER"

    headers["User-Agent"] = "FINDER/1.0"
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST", headers=headers)
    return urllib.request.urlopen(req, timeout=90 if think else 60)


class FinderHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, status: int, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        return self.rfile.read(length) if length > 0 else b""

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/api/chat":
            self.handle_auto_chat()
        elif path == "/api/cloudflare":
            self.handle_cloudflare_chat()
        elif path == "/api/generate-image":
            self.handle_generate_image()
        elif path == "/api/generate-doc":
            self.handle_generate_doc()
        elif path == "/api/study":
            self.handle_study()
        elif path == "/api/study/upload":
            self.handle_study_upload()
        else:
            self.send_json(404, {"error": {"message": "Endpoint não encontrado."}})

    def handle_auto_chat(self):
        """Stream chat with automatic provider/model failover when quota, rate-limit, or availability errors occur."""
        try:
            data = json.loads(self.read_body().decode("utf-8") or "{}")
        except Exception:
            self.send_json(400, {"error": {"message": "JSON inválido."}})
            return

        requested = data.get("provider", "auto")
        if requested != "auto" and requested not in PROVIDERS:
            requested = "auto"
        think = bool(data.get("think", False))
        messages = data.get("messages") or []
        sys_instr = str(data.get("systemInstruction") or "").strip()
        client_settings = data.get("settings") or {}
        account_id = clean_account_id((client_settings.get("cloudflare") or {}).get("accountId"))

        if requested == "auto":
            last_user = next((str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), "")
            is_study_or_long = len(last_user) > 1500 or bool(
                re.search(r"\b(resum[ao]|resumir|expli(que|car)|aula|artigo|estudo|simulado|flashcard)\b", last_user, re.I)
            )
            if think:
                order = THINK_ORDER
            elif is_study_or_long:
                order = ["cohere", "gemini", "groq", "cloudflare", "openrouter"]
            else:
                order = AUTO_ORDER
        else:
            order = THINK_ORDER if think else ([requested] + [p for p in FAILOVER_ORDER if p != requested])

        upstream = None
        chosen = order[0]
        chosen_model = PROVIDERS[chosen]["model"]
        last_error = "Nenhum provedor disponível no momento."

        for idx, prov_id in enumerate(order):
            cfg = client_settings.get(prov_id) or {}
            key = str(cfg.get("key") or "").strip() or PROVIDERS[prov_id]["key"]
            model = str(cfg.get("model") or "").strip() or PROVIDERS[prov_id]["model"]
            if think and prov_id == "cloudflare":
                model = "@cf/openai/gpt-oss-120b"
            if not key:
                continue
            try:
                upstream = open_chat_upstream(prov_id, model, key, messages, sys_instr, account_id, think=think)
                chosen = prov_id
                chosen_model = model
                is_fallback = idx > 0
                break
            except urllib.error.HTTPError as err:
                err_txt = err.read().decode("utf-8", errors="replace")[:200]
                last_error = f"{PROVIDERS[prov_id]['name']} ({err.code}): {err_txt}"
                continue
            except Exception as err:
                last_error = f"{PROVIDERS[prov_id]['name']}: {err}"
                continue

        if upstream is None:
            self.send_json(503, {"error": {"message": f"Todos os provedores falharam. Último erro: {last_error}"}})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.end_headers()

        if chosen != requested:
            meta = json.dumps({
                "finder_switched": {
                    "from": requested,
                    "to": chosen,
                    "mode": "auto" if requested == "auto" else "failover",
                    "fallback": bool(is_fallback),
                    "providerName": PROVIDERS[chosen]["name"],
                    "model": chosen_model,
                }
            })
            self.wfile.write(f"data: {meta}\n\n".encode("utf-8"))
            self.wfile.flush()

        try:
            for raw_line in upstream:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except Exception:
                    continue

                if chosen == "gemini":
                    parts = (((chunk.get("candidates") or [{}])[0]).get("content") or {}).get("parts") or []
                    for p in parts:
                        if not p.get("thought") and isinstance(p.get("text"), str) and p["text"]:
                            out = json.dumps({"choices": [{"delta": {"content": p["text"]}}]})
                            self.wfile.write(f"data: {out}\n\n".encode("utf-8"))
                            self.wfile.flush()
                elif chosen == "cohere":
                    if chunk.get("type") == "content-delta":
                        msg_delta = (chunk.get("delta") or {}).get("message") or {}
                        c = msg_delta.get("content") or {}
                        text = c.get("text", "")
                        if text:
                            out = json.dumps({"choices": [{"delta": {"content": text}}]})
                            self.wfile.write(f"data: {out}\n\n".encode("utf-8"))
                            self.wfile.flush()
                else:
                    delta = ((chunk.get("choices") or [{}])[0]).get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        out = json.dumps({"choices": [{"delta": {"content": content}}]})
                        self.wfile.write(f"data: {out}\n\n".encode("utf-8"))
                        self.wfile.flush()
        finally:
            upstream.close()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    def handle_cloudflare_chat(self):
        body = self.read_body()
        account_id = clean_account_id(self.headers.get("x-cf-account"))
        token = resolve_token(self.headers.get("Authorization"), CF_API_TOKEN)

        req = urllib.request.Request(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as upstream:
                self.send_response(upstream.status)
                self.send_header("Content-Type", upstream.headers.get("Content-Type", "text/event-stream; charset=utf-8"))
                self.end_headers()
                while True:
                    chunk = upstream.read(1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as err:
            status, msg = format_cf_error(err.code, err.read())
            self.send_json(status, {"error": {"message": msg}})
        except Exception as err:
            self.send_json(502, {"error": {"message": f"Erro de conexão com a Cloudflare: {err}"}})

    def handle_generate_image(self):
        try:
            data = json.loads(self.read_body().decode("utf-8") or "{}")
        except Exception:
            self.send_json(400, {"error": {"message": "JSON inválido."}})
            return

        raw_prompt = str(data.get("prompt", "")).strip()
        if not raw_prompt:
            self.send_json(400, {"error": {"message": "Informe uma descrição para gerar a imagem."}})
            return

        account_id = clean_account_id(self.headers.get("x-cf-account") or data.get("accountId"))
        token = resolve_token(self.headers.get("Authorization"), CF_API_TOKEN)
        requested_model = str(data.get("model") or "").strip()
        models_to_try = ([requested_model] if requested_model else []) + [m for m in IMAGE_MODELS if m != requested_model]
        steps = max(1, min(int(data.get("steps") or 4), 8))

        prepared_prompt = translate_prompt_to_english(raw_prompt, account_id, token)
        last_status, last_msg = 502, "Não foi possível gerar a imagem."

        for model in models_to_try:
            payload = json.dumps({"prompt": prepared_prompt, "steps": steps}).encode("utf-8")
            req = urllib.request.Request(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}",
                data=payload,
                method="POST",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as upstream:
                    ctype = (upstream.headers.get("Content-Type") or "").lower()
                    raw = upstream.read()

                    if "application/json" in ctype:
                        parsed = json.loads(raw.decode("utf-8"))
                        b64 = (parsed.get("result") or {}).get("image", "")
                        if not b64:
                            continue
                        mime = "image/png" if b64.startswith("iVBOR") else "image/jpeg"
                    else:
                        mime = "image/png" if "png" in ctype else "image/jpeg"
                        b64 = base64.b64encode(raw).decode("ascii")

                    self.send_json(200, {
                        "image": f"data:{mime};base64,{b64}",
                        "mimeType": mime,
                        "model": model,
                        "promptUsed": prepared_prompt,
                    })
                    return
            except urllib.error.HTTPError as err:
                last_status, last_msg = format_cf_error(err.code, err.read())
                if last_status == 429:
                    break
                continue
            except TimeoutError:
                last_status, last_msg = 504, "A geração da imagem demorou mais que o esperado. Tente novamente."
            except Exception as err:
                last_status, last_msg = 502, f"Erro ao gerar imagem: {err}"

        self.send_json(last_status, {"error": {"message": last_msg}})

    def handle_generate_doc(self):
        """Generate a downloadable PDF or DOCX file from markdown/text content."""
        try:
            data = json.loads(self.read_body().decode("utf-8") or "{}")
        except Exception:
            self.send_json(400, {"error": {"message": "JSON inválido."}})
            return

        fmt = str(data.get("format", "pdf")).strip().lower()
        if fmt not in ("pdf", "docx"):
            fmt = "pdf"
        title = str(data.get("title") or "Documento FINDER").strip()[:100]
        content = str(data.get("content") or "").strip()
        if not content:
            self.send_json(400, {"error": {"message": "Conteúdo vazio para gerar documento."}})
            return

        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", title).strip("_") or "documento"
        filename = f"{safe_name}.{fmt}"

        try:
            if fmt == "pdf":
                raw_bytes = build_pdf_bytes(title, content)
                mime = "application/pdf"
            else:
                raw_bytes = build_docx_bytes(title, content)
                mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

            b64 = base64.b64encode(raw_bytes).decode("ascii")
            self.send_json(200, {
                "ok": True,
                "filename": filename,
                "mimeType": mime,
                "dataUrl": f"data:{mime};base64,{b64}",
            })
        except Exception as err:
            self.send_json(500, {"error": {"message": f"Erro ao gerar {fmt.upper()}: {err}"}})

    def handle_study(self):
        """Process study mode requests using Cohere."""
        try:
            data = json.loads(self.read_body().decode("utf-8") or "{}")
        except Exception:
            self.send_json(400, {"error": {"message": "JSON inválido."}})
            return

        action = str(data.get("action", "")).strip()
        subject = str(data.get("subject", "")).strip()
        content = str(data.get("content", "")).strip()
        file_id = str(data.get("fileId", "")).strip()
        history = data.get("history") or []

        if not action:
            self.send_json(400, {"error": {"message": "Informe uma ação (lesson, summary, quiz, flashcards, explain, research, chat)."}})
            return

        # Rate limit check
        rate_err = _check_study_rate()
        if rate_err:
            self.send_json(429, {"error": {"message": rate_err}})
            return

        try:
            result = {}

            if action == "research":
                if not subject:
                    self.send_json(400, {"error": {"message": "Informe o assunto para pesquisar."}})
                    return
                articles = search_academic(subject, max_results=5)
                articles_text = "\n".join(
                    f"- {a['title']} ({a['year']}) DOI: {a['doi']}" + (f"\n  Resumo: {a['abstract'][:300]}..." if a.get('abstract') else "")
                    for a in articles
                )
                prompt = STUDY_PROMPTS["research"].format(subject=subject, articles=articles_text)
                text = cohere_chat([{"role": "user", "content": prompt}])
                result = {"text": text, "articles": articles}

            elif action == "chat":
                if not history and not subject:
                    self.send_json(400, {"error": {"message": "Envie pelo menos uma mensagem."}})
                    return
                msgs = []
                msgs.append({"role": "system", "content":
                    "Você é o assistente de estudos do FINDER. Ajude o usuário a aprender, explique conceitos, "
                    "tire dúvidas e sugira materiais de estudo. Seja didático, claro e encoraje o aprendizado. "
                    "Se o usuário enviar um texto ou trecho de documento, ajude a interpretá-lo."
                })
                for m in history[-20:]:  # keep last 20 messages for context
                    if m.get("role") in ("user", "assistant") and m.get("content"):
                        msgs.append({"role": m["role"], "content": m["content"]})
                if subject:
                    msgs.append({"role": "user", "content": subject})

                # Stream this response
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.end_headers()

                key = COHERE_API_KEY
                body_bytes = json.dumps({"model": "command-a-plus-05-2026", "messages": msgs, "stream": True}).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.cohere.com/v2/chat",
                    data=body_bytes,
                    method="POST",
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": "FINDER/1.0"},
                )
                upstream = urllib.request.urlopen(req, timeout=60)
                try:
                    for raw_line in upstream:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(payload)
                        except Exception:
                            continue
                        if chunk.get("type") == "content-delta":
                            msg_delta = (chunk.get("delta") or {}).get("message") or {}
                            c = msg_delta.get("content") or {}
                            text = c.get("text", "")
                            if text:
                                out = json.dumps({"choices": [{"delta": {"content": text}}]})
                                self.wfile.write(f"data: {out}\n\n".encode("utf-8"))
                                self.wfile.flush()
                finally:
                    upstream.close()
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                return

            elif action in ("lesson", "quiz", "flashcards", "explain"):
                if not subject:
                    self.send_json(400, {"error": {"message": "Informe o assunto."}})
                    return
                context = ""
                if file_id:
                    filepath = os.path.join(STUDY_DIR, file_id)
                    if os.path.exists(filepath):
                        if filepath.lower().endswith(".pdf"):
                            context = extract_pdf_text(filepath)[:8000]
                        elif filepath.lower().endswith(".docx"):
                            context = extract_docx_text(filepath)[:8000]
                        else:
                            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                                context = f.read()[:8000]
                elif content:
                    context = content[:8000]

                prompt = STUDY_PROMPTS[action].format(subject=subject, context=context)
                text = cohere_chat([{"role": "user", "content": prompt}])
                result = {"text": text}

            elif action == "summary":
                text_to_summarize = content
                if not text_to_summarize and file_id:
                    filepath = os.path.join(STUDY_DIR, file_id)
                    if os.path.exists(filepath):
                        if filepath.lower().endswith(".pdf"):
                            text_to_summarize = extract_pdf_text(filepath)[:12000]
                        elif filepath.lower().endswith(".docx"):
                            text_to_summarize = extract_docx_text(filepath)[:12000]
                        else:
                            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                                text_to_summarize = f.read()[:12000]

                if not text_to_summarize:
                    self.send_json(400, {"error": {"message": "Informe o conteúdo ou envie um arquivo para resumir."}})
                    return

                prompt = STUDY_PROMPTS["summary"].format(content=text_to_summarize)
                text = cohere_chat([{"role": "user", "content": prompt}])
                result = {"text": text}

            else:
                self.send_json(400, {"error": {"message": f"Ação desconhecida: {action}"}})
                return

            self.send_json(200, {"ok": True, **result})

        except urllib.error.HTTPError as err:
            status = err.code
            body = err.read().decode("utf-8", errors="replace")[:300]
            if status == 429:
                self.send_json(429, {"error": {"message": "Limite Cohere atingido. Aguarde um momento."}})
            else:
                self.send_json(status, {"error": {"message": f"Erro Cohere ({status}): {body}"}})
        except Exception as err:
            self.send_json(502, {"error": {"message": f"Erro no Modo Estudo: {err}"}})

    def handle_study_upload(self):
        """Handle file uploads for study mode (PDF, DOCX, TXT)."""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_json(400, {"error": {"message": "Envie o arquivo como multipart/form-data."}})
            return

        try:
            import cgi
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type}
            )
            file_item = form["file"]
            if not file_item.filename:
                self.send_json(400, {"error": {"message": "Nenhum arquivo enviado."}})
                return

            ext = os.path.splitext(file_item.filename)[1].lower()
            if ext not in (".pdf", ".docx", ".txt", ".md"):
                self.send_json(400, {"error": {"message": "Formato não suportado. Envie PDF, DOCX, TXT ou MD."}})
                return

            import time
            file_id = f"{int(time.time())}_{re.sub(r'[^a-zA-Z0-9._-]', '_', file_item.filename)}"
            filepath = os.path.join(STUDY_DIR, file_id)

            with open(filepath, "wb") as f:
                f.write(file_item.file.read())

            # Extract preview text
            preview = ""
            if ext == ".pdf":
                preview = extract_pdf_text(filepath)[:500]
            elif ext == ".docx":
                preview = extract_docx_text(filepath)[:500]
            else:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    preview = f.read()[:500]

            self.send_json(200, {
                "ok": True,
                "fileId": file_id,
                "filename": file_item.filename,
                "preview": preview,
            })
        except Exception as err:
            self.send_json(500, {"error": {"message": f"Erro ao processar arquivo: {err}"}})


# ════════════════════════════════════════════
#  STUDY MODE — Helpers
# ════════════════════════════════════════════

STUDY_DIR = os.path.join(ROOT_DIR, "study_uploads")
os.makedirs(STUDY_DIR, exist_ok=True)

_study_rate = {"count": 0, "minute": 0}  # simple per-minute rate limiter
COHERE_RPM = 18  # stay under 20 rpm trial limit


def _check_study_rate() -> str | None:
    """Return error message if rate limit exceeded, else None."""
    import time
    minute = int(time.time()) // 60
    if _study_rate["minute"] != minute:
        _study_rate["minute"] = minute
        _study_rate["count"] = 0
    if _study_rate["count"] >= COHERE_RPM:
        return "Limite de requisições Cohere atingido (20/min). Aguarde um momento."
    _study_rate["count"] += 1
    return None


def cohere_chat(messages: list, model: str = "command-a-plus-05-2026", temperature: float = 0.3) -> str:
    """Call Cohere v2 chat (non-streaming) and return the text content."""
    key = COHERE_API_KEY
    if not key:
        raise RuntimeError("Chave da Cohere não configurada.")
    body = json.dumps({"model": model, "messages": messages, "temperature": temperature}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.cohere.com/v2/chat",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": "FINDER/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    parts = data.get("message", {}).get("content", [])
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


def search_academic(query: str, max_results: int = 5) -> list[dict]:
    """Search OpenAlex for academic works."""
    q = urllib.parse.quote(query)
    url = f"https://api.openalex.org/works?search={q}&per-page={max_results}"
    req = urllib.request.Request(url, headers={"User-Agent": "FINDER-Academic/1.0 (mailto:finder@local.app)"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = []
        for w in data.get("results", []):
            results.append({
                "title": w.get("title", ""),
                "year": w.get("publication_year"),
                "doi": w.get("doi", ""),
                "cited_by": w.get("cited_by_count", 0),
                "abstract": (w.get("abstract_inverted_index") or None) and _reconstruct_abstract(w["abstract_inverted_index"]),
            })
        return results
    except Exception:
        return []


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted index."""
    if not inverted_index:
        return ""
    words: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[k] for k in sorted(words))


def extract_pdf_text(filepath: str) -> str:
    """Extract text from a PDF file."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
        return "\n\n".join(text)
    except Exception as e:
        return f"[Erro ao extrair PDF: {e}]"


def extract_docx_text(filepath: str) -> str:
    """Extract text from a DOCX file."""
    try:
        from docx import Document
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"[Erro ao extrair DOCX: {e}]"


# ── Study action prompts ──

STUDY_PROMPTS = {
    "lesson": (
        "Crie um material de estudo completo e aprofundado sobre o assunto a seguir. "
        "Organize em:\n1. Introdução\n2. Explicação aprofundada\n3. Conceitos fundamentais\n"
        "4. Exemplos práticos\n5. Resumo\n\nAssunto: {subject}\n\n{context}"
    ),
    "summary": (
        "Resuma o seguinte conteúdo de forma clara e organizada, destacando os pontos mais importantes:\n\n{content}"
    ),
    "quiz": (
        "Gere um simulado com 10 questões de múltipla escolha sobre o assunto a seguir. "
        "Para cada questão, forneça 4 alternativas (A, B, C, D), indique a resposta correta e "
        "explique brevemente por que é a correta.\n\nAssunto: {subject}\n\n{context}"
    ),
    "flashcards": (
        "Gere 10 flashcards (cartões de estudo) sobre o assunto a seguir. "
        "Retorne em formato:\n\n**Frente:** [pergunta]\n**Verso:** [resposta]\n\n"
        "para cada flashcard.\n\nAssunto: {subject}\n\n{context}"
    ),
    "explain": (
        "Explique o seguinte assunto passo a passo, como se estivesse ensinando para alguém que está "
        "aprendendo do zero. Use exemplos simples e analogias.\n\nAssunto: {subject}\n\n{context}"
    ),
    "research": (
        "Com base nos artigos acadêmicos encontrados abaixo, crie uma síntese organizada sobre o assunto. "
        "Inclua:\n1. Visão geral do tema\n2. Principais descobertas\n3. Conceitos relevantes\n"
        "4. Referências dos artigos\n\nAssunto: {subject}\n\nArtigos encontrados:\n{articles}"
    ),
}


def build_pdf_bytes(title: str, content: str) -> bytes:
    """Build a UTF-8 PDF document from markdown/plain text."""
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    reg_font = "C:/Windows/Fonts/arial.ttf"
    bold_font = "C:/Windows/Fonts/arialbd.ttf"
    if os.path.exists(reg_font):
        pdf.add_font("DocFont", "", reg_font)
        pdf.add_font("DocFont", "B", bold_font if os.path.exists(bold_font) else reg_font)
        font_family = "DocFont"
    else:
        font_family = "Helvetica"

    def sanitize(txt: str) -> str:
        # Remove 4-byte emojis/symbols unsupported by Arial TTF
        return re.sub(r"[\U00010000-\U0010ffff]", "", txt)

    def write_block(txt: str, size: int = 11, bold: bool = False, line_h: float = 6.5, space_after: float = 0):
        clean_txt = sanitize(txt).strip()
        if not clean_txt:
            return
        pdf.set_x(pdf.l_margin)
        pdf.set_font(font_family, "B" if bold else "", size)
        try:
            pdf.multi_cell(pdf.epw, line_h, clean_txt, new_x="LMARGIN", new_y="NEXT")
        except Exception:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, line_h, clean_txt, wrapmode="CHAR", new_x="LMARGIN", new_y="NEXT")
        if space_after:
            pdf.ln(space_after)

    write_block(title, size=16, bold=True, line_h=9, space_after=3)

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            pdf.ln(3)
            continue
        if re.fullmatch(r"[-*_]{3,}", line):
            pdf.ln(2)
            continue
        clean = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        clean = re.sub(r"`(.*?)`", r"\1", clean)
        if clean.startswith("### "):
            write_block(clean[4:], size=12, bold=True, line_h=7, space_after=1)
        elif clean.startswith("## "):
            write_block(clean[3:], size=13, bold=True, line_h=8, space_after=1)
        elif clean.startswith("# "):
            write_block(clean[2:], size=15, bold=True, line_h=8, space_after=2)
        elif clean.startswith(("- ", "* ")):
            write_block("• " + clean[2:], size=11, bold=False, line_h=6.5)
        else:
            write_block(clean, size=11, bold=False, line_h=6.5)

    return bytes(pdf.output())


def build_docx_bytes(title: str, content: str) -> bytes:
    """Build a DOCX document from markdown/plain text."""
    from docx import Document
    doc = Document()
    doc.add_heading(title, level=0)

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        clean = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        clean = re.sub(r"`(.*?)`", r"\1", clean)
        if clean.startswith("### "):
            doc.add_heading(clean[4:], level=3)
        elif clean.startswith("## "):
            doc.add_heading(clean[3:], level=2)
        elif clean.startswith("# "):
            doc.add_heading(clean[2:], level=1)
        elif clean.startswith(("- ", "* ")):
            doc.add_paragraph(clean[2:], style="List Bullet")
        else:
            doc.add_paragraph(clean)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


if __name__ == "__main__":
    ThreadingHTTPServer.allow_reuse_address = False
    server = ThreadingHTTPServer((HOST, PORT), FinderHandler)
    print(f"FINDER Python server running at http://localhost:{PORT}", flush=True)
    server.serve_forever()
