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

with open(Path(__file__).parent / "jc_data.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    CAPTION_TYPE_MAP = config["caption_type_map"]
    EXTRA_OPTIONS = config["extra_options"]
    MODEL_SETTINGS = config["model_settings"]
    CAPTION_LENGTH_CHOICES = config["caption_length_choices"]
    GGUF_MODELS = config["gguf_models"]

_MODEL_CACHE = None
_MODEL_CACHE_LOCK = None

_CHAT_HANDLER_CACHE = {}
_CHAT_HANDLER_CACHE_LOCK = None
try:
    import threading as _t
    _CHAT_HANDLER_CACHE_LOCK = _t.RLock()
except Exception:
    _CHAT_HANDLER_CACHE_LOCK = None

def _make_chat_handler(mmproj_path: Path, cache: bool = False):
    return Llava15ChatHandler(clip_model_path=str(mmproj_path.resolve()))

def _drop_all_chat_handlers():
    try:
        it = list(_CHAT_HANDLER_CACHE.items())
    except Exception:
        it = []
    for _, _ch in it:
        for _m in ("close", "free", "__del__"):
            try:
                getattr(_ch, _m)()
            except Exception:
                pass
    try:
        _CHAT_HANDLER_CACHE.clear()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass

def _purge_cached_key(strong_key: str):
    return

def _purge_all_cached():
    return
try:
    import atexit
    atexit.register(_purge_all_cached)
    atexit.register(_drop_all_chat_handlers)
except Exception:
    pass

def _strong_key_from(model: str, processing_mode: str) -> str:
    model_name = GGUF_MODELS[model]["name"]
    model_filename = Path(model_name).name
    return f"{model}_{processing_mode}|{model_filename}|ctx{MODEL_SETTINGS['context_window']}"

def _free_llama_model(llm):
    try:
        if hasattr(llm, "close") and callable(llm.close):
            llm.close()
        elif hasattr(llm, "free") and callable(llm.free):
            llm.free()
    except Exception:
        pass

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
    state = {"model_path": None, "mmproj_path": None, "processing_mode": "GPU"}

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
                    "processing_mode": msg.get("processing_mode","GPU"),
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
                    numa=(state["processing_mode"] == "CPU"),
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
    """모델/모드별로 하나의 상주 워커 유지. 격리는 유지하고 재로딩 누수/지연 제거."""
    def __init__(self):
        self.proc = None
        self.stdin = None
        self.stdout = None
        self.sig = None

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

    def _send(self, obj):
        self.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.stdin.flush()

    def _recv(self, timeout=120.0):
        end = time.time() + timeout
        noise = []
        while time.time() < end:
            line = self.stdout.readline()
            if not line:
                rc = self.proc.poll()
                if rc is not None:
                    tail = ""
                    try:
                        tail = (self.proc.stderr.read() or "")[-2000:]
                    except Exception:
                        pass
                    raise RuntimeError(f"worker exit rc={rc} stderr_tail={tail}")
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
        tail = ""
        try:
            tail = (self.proc.stderr.read() or "")[-2000:]
        except Exception:
            pass
        raise RuntimeError("worker recv timeout; noise=" + " | ".join(noise) + f" | stderr_tail={tail}")

    def ensure(self, *, model_path, mmproj_path, processing_mode, n_ctx, n_batch, n_threads, n_gpu_layers):
        want = (model_path, mmproj_path, processing_mode)
        if self.proc is None or self.proc.poll() is not None or self.sig != want:
            self.close()
            self._spawn()
            self._send({
                "cmd": "init",
                "model_path": model_path,
                "mmproj_path": mmproj_path,
                "processing_mode": processing_mode,
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
        txt = resp.get("text", "") or ""
        try:
            cut_pos = min([p for m in cut_markers for p in [txt.find(m)] if p >= 0], default=-1)
            if cut_pos >= 0:
                txt = txt[:cut_pos].rstrip()
        except Exception:
            pass
        return txt

    def close(self):
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
        parts = model.split("/")
        if len(parts) < 3:
            if len(parts) != 2:
                raise ValueError("Invalid model path")
            repo_path = "/".join(parts[:2])
            filename = parts[-1]
        else:
            repo_path = "/".join(parts[:2])
            filename  = "/".join(parts[2:])
            repo_id=repo_path,
            filename=filename,
            local_dir=str(llm_models_dir),
            local_dir_use_symlinks=False
        )).resolve()

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

from typing import Union, List
_LEN_MAP = {
    "very short": 10, "short": 20, "medium": 40, "long": 80, "very long": 120
}
def build_prompt(caption_type: str, caption_length: Union[str, int], extra_options: List[str], name_input: str) -> str:
    if isinstance(caption_type, str) and caption_type.lower() == "none":
        return ""
    if caption_length == "any":
        map_idx = 0
    elif isinstance(caption_length, str) and caption_length.isdigit():
        map_idx = 1
    else:
        map_idx = 2
    
    prompt = CAPTION_TYPE_MAP[caption_type][map_idx]
    tmpl = CAPTION_TYPE_MAP[caption_type][0]
    if isinstance(caption_length, int):
        wc = max(1, caption_length)
    elif isinstance(caption_length, str):
        s = caption_length.strip().lower()
        wc = _LEN_MAP.get(s, 0)
    else:
        wc = 0
    prompt = tmpl
    if extra_options:
        prompt += " " + " ".join(extra_options)
    return prompt.format(name=name_input or "{NAME}", length=caption_length, word_count=caption_length)

class JC_GGUF_adv:
    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(GGUF_MODELS.keys())
        style_list = list(CAPTION_TYPE_MAP.keys()) + ["None"]
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (model_list, {"default": model_list[0], "tooltip": "Select the GGUF model to use for caption generation"}),
                "processing_mode": (["GPU", "CPU"], {"default": "GPU", "tooltip": "GPU: Faster but requires more VRAM\nCPU: Slower but saves VRAM"}),
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
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("PROMPT", "STRING")
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
    
    def generate(self, image, model, processing_mode, prompt_style, caption_length, max_new_tokens, temperature, top_p, top_k, custom_prompt, seed=-1, extra_options=None):
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
                self._wm.ensure(
                    model_path=model_path,
                    mmproj_path=mmproj_path,
                    processing_mode=processing_mode,
                    n_ctx=MODEL_SETTINGS["context_window"],
                    n_batch=1536,
                    n_threads=max(4, MODEL_SETTINGS["cpu_threads"]),
                    n_gpu_layers=(-1 if processing_mode == "GPU" else 0),
                )
                response = self._wm.infer(
                    image_path=tmp_png,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=used_seed,
                    stop=stop_tokens,
                    cut_markers=stop_tokens,
                )

            return (prompt, response)
        except Exception as e:
            return ("", f"Error: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "JC_GGUF_adv": JC_GGUF_adv,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JC_GGUF_adv": "JoyCaption GGUF (Advanced)",
}



