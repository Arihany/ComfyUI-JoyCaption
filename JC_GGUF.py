import torch
import folder_paths
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToPILImage
import json
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import multiprocessing as mp
import base64
import io
import gc
import os
import sys
import subprocess
import tempfile
import shutil
import time
from huggingface_hub import hf_hub_download
import secrets
import random
from JC import JC_ExtraOptions
import threading
from collections import deque
from typing import Union, List

with open(Path(__file__).parent / "jc_data.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    CAPTION_TYPE_MAP = config["caption_type_map"]
    EXTRA_OPTIONS = config["extra_options"]
    MODEL_SETTINGS = config["model_settings"]
    CAPTION_LENGTH_CHOICES = config["caption_length_choices"]
    GGUF_MODELS = config["gguf_models"]

_CHAT_HANDLER_CACHE = {}

def _make_chat_handler(mmproj_path: Path, cache: bool = False):
    return Llava15ChatHandler(clip_model_path=str(mmproj_path.resolve()))

_WORKER_SCRIPT = r"""# -*- coding: utf-8 -*-
import os, sys, json, base64, traceback, pathlib

def jprint(obj):
    sys.stdout.write("J:" + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="strict")
except Exception:
    pass

try:
    if os.name == "nt":
        dll_dir = os.environ.get("LLAMA_DLL_DIR", "")
        if dll_dir and os.path.isdir(dll_dir):
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        cuda_root = os.environ.get("CUDA_PATH", "")
        if cuda_root:
            cuda_bin = os.path.join(cuda_root, "bin")
            if os.path.isdir(cuda_bin):
                os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

try:
    from llama_cpp import Llama, llama_backend_free
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except Exception as e:
    jprint({"ok": False, "phase": "import", "error": str(e), "trace": traceback.format_exc()[-1200:]})
    sys.exit(1)

def load_image_as_data_uri(path):
    with open(path, "rb") as f:
        b = f.read()
    return "data:image/png;base64," + base64.b64encode(b).decode("utf-8")

def main():
    os.environ["LLAMA_LOG_DISABLE"] = "1"
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    os.environ["LLAMA_CLIP_N_GPU_LAYERS"] = "0"

    llm = None
    chat = None
    state = {"model_path": None, "mmproj_path": None}

    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        cmd = msg.get("cmd")
        try:
            if cmd == "init":
                if llm is not None:
                    try: llm.chat_handler = None
                    except: pass
                    try: llm.close()
                    except: pass
                    llm = None
                if chat is not None:
                    try:
                        for m in ("close","free","__del__"):
                            getattr(chat, m)()
                    except Exception:
                        pass
                    chat = None
                state.update({
                    "model_path": msg["model_path"],
                    "mmproj_path": msg["mmproj_path"],
                })
                chat = Llava15ChatHandler(clip_model_path=state["mmproj_path"])
                llm = Llama(
                    model_path=state["model_path"],
                    n_ctx=msg["n_ctx"],
                    n_batch=msg["n_batch"],
                    n_threads=msg["n_threads"],
                    n_gpu_layers=msg["n_gpu_layers"],
                    verbose=False,
                    chat_handler=chat,
                    offload_kqv=True,
                )
                jprint({"ok": True, "ready": True})

            elif cmd == "infer":
                if llm is None:
                    jprint({"ok": False, "error": "not_initialized"})
                    continue
                data_uri = load_image_as_data_uri(msg["image_path"])
                messages = []
                sys_txt = (msg.get("system_prompt","") or "").strip()
                if sys_txt:
                    messages.append({"role":"system","content": sys_txt})
                user_txt = (msg.get("prompt","") or "").strip()
                if not user_txt:
                    user_txt = " "
                messages.append({
                    "role":"user",
                    "content":[
                        {"type":"text","text": user_txt},
                        {"type":"image_url","image_url":{"url": data_uri}}
                    ]
                })
                params = dict(
                    messages=messages,
                    max_tokens=msg["max_new_tokens"],
                    temperature=msg["temperature"],
                    top_p=msg["top_p"],
                    stream=False,
                    repeat_penalty=1.1,
                    mirostat_mode=0,
                    stop=msg["stop"],
                )
                if msg.get("top_k",0) > 0:
                    params["top_k"] = msg["top_k"]
                if "seed" in msg:
                    try:
                        params["seed"] = int(msg["seed"])
                    except Exception:
                        params["seed"] = 0
                resp = llm.create_chat_completion(**params)
                text = (resp["choices"][0]["message"]["content"] or "").strip()
                for cut in msg["cut_markers"]:
                    p = text.find(cut)
                    if p != -1:
                        text = text[:p].rstrip()
                jprint({"ok": True, "text": text})

            elif cmd == "free":
                if llm is not None:
                    try: llm.chat_handler = None
                    except: pass
                    try: llm.close()
                    except: pass
                    llm = None
                if chat is not None:
                    try:
                        for m in ("close","free","__del__"):
                            getattr(chat, m)()
                    except Exception:
                        pass
                    chat = None
                jprint({"ok": True})

            elif cmd == "shutdown":
                break
            else:
                jprint({"ok": False, "error": "unknown_command"})
        except Exception as e:
            jprint({"ok": False, "error": str(e)})

    try:
        if llm is not None:
            try: llm.chat_handler = None
            except: pass
            try: llm.close()
            except: pass
    except: pass
    try: llama_backend_free()
    except: pass

if __name__ == "__main__":
    main()
"""

class _WorkerManager:
    def __init__(self):
        self.proc = None
        self.stdin = None
        self.stdout = None
        self.sig = None
        self._lock = threading.RLock()
        self._stderr_tail = deque(maxlen=2000)
        self._stderr_thread = None
        self._tmpdir = None
        self._timeout = float(MODEL_SETTINGS.get("worker_timeout", 300.0))

    def _spawn(self):
        tmpdir = tempfile.mkdtemp(prefix="jcgguf_srv_")
        worker_py = os.path.join(tmpdir, "worker.py")
        with open(worker_py, "w", encoding="utf-8") as f:
            f.write(_WORKER_SCRIPT)
        env = os.environ.copy()
        try:
            import llama_cpp as _llcp
            from pathlib import Path as _P
            _libdir = str((_P(_llcp.__file__).resolve().parent / "lib"))
            if os.path.isdir(_libdir):
                env["LLAMA_DLL_DIR"] = _libdir
                env["PATH"] = _libdir + os.pathsep + env.get("PATH", "")
        except Exception:
            pass
        env.update({
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "LLAMA_LOG_DISABLE": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        })
        self.proc = subprocess.Popen(
            [sys.executable, "-u", worker_py],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env
        )
        self.stdin = self.proc.stdin
        self.stdout = self.proc.stdout
        self._tmpdir = tmpdir
        def _drain():
            try:
                for line in self.proc.stderr:
                    if not line:
                        break
                    s = line.strip()
                    if s:
                        self._stderr_tail.extend(s[-2000:])
            except Exception:
                pass
        self._stderr_thread = threading.Thread(target=_drain, name="jcgguf-stderr", daemon=True)
        self._stderr_thread.start()

    def _send(self, obj):
        try:
            self.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.stdin.flush()
        except Exception as e:
            raise RuntimeError(f"worker send failed: {e}")

    def _recv(self, timeout=None):
        if timeout is None:
            timeout = self._timeout
        end = time.time() + float(timeout)
        noise = []
        while time.time() < end:
            line = self.stdout.readline()
            if not line:
                rc = self.proc.poll()
                if rc is not None:
                    tail = "".join(self._stderr_tail)
                    raise RuntimeError(f"worker exit rc={rc} stderr_tail={tail[-2000:]}")
                time.sleep(0.01)
                continue
            s = line.strip()
            if not s:
                continue
            if not s.startswith("J:"):
                if len(noise) < 5:
                    noise.append(s[:240])
                continue
            payload = s[2:]
            try:
                return json.loads(payload)
            except Exception:
                if len(noise) < 5:
                    noise.append(("badjson:" + payload)[:240])
                continue
        tail = "".join(self._stderr_tail)
        raise RuntimeError("worker recv timeout; noise=" + " | ".join(noise) + f" | stderr_tail={tail[-2000:]}")

    def ensure(self, *, model_path, mmproj_path, n_ctx, n_batch, n_threads, n_gpu_layers):
        with self._lock:
            want = (model_path, mmproj_path)
            if self.proc is None or self.proc.poll() is not None or self.sig != want:
                self.close()
                self._spawn()
                self._send({
                    "cmd": "init",
                    "model_path": model_path,
                    "mmproj_path": mmproj_path,
                    "n_ctx": n_ctx,
                    "n_batch": n_batch,
                    "n_threads": n_threads,
                    "n_gpu_layers": n_gpu_layers,
                })
                resp = self._recv()
                if not (isinstance(resp, dict) and resp.get("ok") and resp.get("ready")):
                    raise RuntimeError(f"worker init failed: {resp}")
                self.sig = want

    def infer(self, *, image_path, system_prompt, prompt, max_new_tokens, temperature, top_p, top_k, seed, stop, cut_markers):
        with self._lock:
            self._send({
                "cmd": "infer",
                "image_path": image_path,
                "system_prompt": system_prompt,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": int(seed),
                "stop": stop,
                "cut_markers": cut_markers,
            })
            resp = self._recv()
            return resp.get("text", "") or ""

    def close(self):
        with self._lock:
            try:
                if self.stdin:
                    try:
                        self._send({"cmd":"shutdown"})
                    except Exception:
                        pass
            finally:
                if self.proc is not None:
                    try: self.proc.kill()
                    except Exception: pass
                self.proc = None
                self.stdin = None
                self.stdout = None
                self.sig = None
                try:
                    shutil.rmtree(getattr(self, "_tmpdir", ""), ignore_errors=True)
                except Exception:
                    pass

def _resolve_paths(model: str) -> tuple[str, str]:
    models_dir = Path(folder_paths.models_dir).resolve()
    llm_models_dir = (models_dir / "LLM" / "GGUF").resolve()
    llm_models_dir.mkdir(parents=True, exist_ok=True)

    model_filename = Path(model).name
    local_path = llm_models_dir / model_filename
    if not local_path.exists():
        parts = model.strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid model spec: {model!r}")
        repo_id = "/".join(parts[:2])
        filename = parts[-1] if len(parts) == 2 else "/".join(parts[2:])
        try:
            local_path = Path(hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(llm_models_dir),
                local_dir_use_symlinks=False,
            )).resolve()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch model from hub: {repo_id}/{filename} :: {e}")

    mmproj_filename = "llama-joycaption-beta-one-llava-mmproj-model-f16.gguf"
    mmproj_path = llm_models_dir / mmproj_filename
    if not mmproj_path.exists():
        mmproj_path = Path(hf_hub_download(
            repo_id="concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf",
            filename=mmproj_filename,
            local_dir=str(llm_models_dir),
            local_dir_use_symlinks=False
        )).resolve()
    return (str(local_path), str(mmproj_path))

_LEN_MAP = {
    "very short": 10, "short": 20, "medium": 40, "long": 80, "very long": 120
}
class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"

def _escape_braces(s: str) -> str:
    # Prevent .format() from choking on arbitrary braces in extras
    return s.replace("{", "{{").replace("}", "}}")

def build_prompt(caption_type: str, caption_length: Union[str, int], extra_options: List[str], name_input: str) -> str:
    if isinstance(caption_type, str) and caption_type.lower() == "none":
        return ""
    if caption_length == "any":
        map_idx = 0
    elif isinstance(caption_length, str) and caption_length.isdigit():
        map_idx = 1
    else:
        map_idx = 2
    
    tmpl = CAPTION_TYPE_MAP[caption_type][map_idx]
    if isinstance(caption_length, int):
        wc = max(1, caption_length)
    elif isinstance(caption_length, str):
        s = caption_length.strip().lower()
        wc = _LEN_MAP.get(s, 0)
    else:
        wc = 0
    extras = ""
    if extra_options:
        extras = " " + " ".join(_escape_braces(x) for x in extra_options)
    return tmpl.format_map(_SafeDict({
        "name": name_input or "{NAME}",
        "length": caption_length,
        "word_count": wc
    })) + extras

class JC_GGUF_adv:
    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(GGUF_MODELS.keys())
        style_list = list(CAPTION_TYPE_MAP.keys()) + ["None"]
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (model_list, {"default": model_list[0], "tooltip": "Select the GGUF model to use for caption generation"}),
                "prompt_style": (style_list, {
                    "default": "Descriptive",
                    "tooltip": "Select the style of caption you want to generate\n'None': no template is applied"
                }),
                "caption_length": (CAPTION_LENGTH_CHOICES, {"default": "any", "tooltip": "Control the length of the generated caption"}),
                "max_new_tokens": ("INT", {"default": MODEL_SETTINGS["default_max_tokens"], "min": 1, "max": 2048, "tooltip": "Maximum number of tokens to generate. Higher values allow longer captions"}),
                "temperature": ("FLOAT", {"default": MODEL_SETTINGS["default_temperature"], "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Control the randomness of the output. Higher values make the output more creative but less predictable"}),
                "top_p": ("FLOAT", {"default": MODEL_SETTINGS["default_top_p"], "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Control the diversity of the output. Higher values allow more diverse word choices"}),
                "top_k": ("INT", {"default": MODEL_SETTINGS["default_top_k"], "min": 0, "max": 100, "tooltip": "Limit the number of possible next tokens. Lower values make the output more focused"}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom prompt template. If empty, will use the selected prompt style"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "ComfyUI-style seed. -1/randomize = random per run; >=0 = deterministic"}),
            },
            "optional": {
                "extra_options": ("JOYCAPTION_EXTRA_OPTIONS", {"tooltip": "Additional options to customize the caption generation"}),
                "kill_worker_after_run": ("BOOLEAN", {"default": False, "tooltip": "On: kill worker process after each run to free ALL VRAM. Off: keep worker for fast consecutive runs"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("CAPTION",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/📝JoyCaption"

    def __init__(self):
        self._wm = _WorkerManager()

    def __del__(self):
        try:
            if hasattr(self, "_wm") and self._wm:
                self._wm.close()
        except Exception:
            pass
    
    def generate(self, image, model, prompt_style, caption_length, max_new_tokens, temperature, top_p, top_k, custom_prompt, seed=-1, extra_options=None, kill_worker_after_run=False):
        try:

            model_path, mmproj_path = _resolve_paths(GGUF_MODELS[model]["name"])

            _style = str(prompt_style).strip().lower()
            if _style in ("none", "off", "raw", "blank"):
                prompt = custom_prompt.strip()
                system_prompt = ""
            else:
                prompt = (custom_prompt if custom_prompt.strip()
                          else build_prompt(
                              prompt_style,
                              caption_length,
                              (extra_options[0] if extra_options and len(extra_options) > 0 else []),
                              (extra_options[1] if extra_options and len(extra_options) > 1 else "{NAME}")
                          ))
                system_prompt = MODEL_SETTINGS["default_system_prompt"]

            with torch.inference_mode():
                pil_image = ToPILImage()(image[0].permute(2, 0, 1).cpu())
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            pil_image = pil_image.resize((336, 336), Image.Resampling.BILINEAR)

            def _coerce_seed(v):
                if isinstance(v, (int, float)):
                    v = int(v)
                    return v if v >= 0 else secrets.randbits(31)
                if isinstance(v, str):
                    s = v.strip().lower()
                    if s in ("", "random", "randomize", "rand", "r", "none", "-1"):
                        return secrets.randbits(31)
                    try:
                        n = int(s, 10)
                        return n if n >= 0 else secrets.randbits(31)
                    except Exception:
                        return secrets.randbits(31)

                return secrets.randbits(31)
            used_seed = _coerce_seed(seed)

            with tempfile.TemporaryDirectory(prefix="jcgguf_") as tmpdir:
                tmp_png = os.path.join(tmpdir, "in.png")
                pil_image.save(tmp_png, format="PNG")
                stop_tokens = [
                    "<|eot_id|>", "</s>",
                    "<|start_header_id|>assistant<|end_header_id|>",
                    "<|start_header_id|>user<|end_header_id|>",
                    "ASSISTANT", "ASSISTANT:", "Assistant:", "Assistant",
                    "USER:", "User:", "USER", "User",
                ]
                cpu_count = os.cpu_count() or 8
                self._wm.ensure(
                    model_path=model_path,
                    mmproj_path=mmproj_path,
                    n_ctx=MODEL_SETTINGS["context_window"],
                    n_batch=1536,
                    n_threads=max(1, min(int(MODEL_SETTINGS["cpu_threads"]), cpu_count)),
                    n_gpu_layers=-1,
                )
                response = self._wm.infer(
                    image_path=tmp_png,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=max(0, int(top_k)),
                    seed=used_seed,
                    stop=stop_tokens,
                    cut_markers=stop_tokens,
                )
                if bool(kill_worker_after_run):
                    try:
                        self._wm.close()
                    except Exception:
                        pass

            return (response,)
        except Exception as e:
            return (f"Error: {str(e)}",)

NODE_CLASS_MAPPINGS = {
    "JC_GGUF_adv": JC_GGUF_adv,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JC_GGUF_adv": "JoyCaption GGUF (Advanced)",
}
