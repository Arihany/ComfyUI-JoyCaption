import torch
import folder_paths
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToPILImage
import json
from llama_cpp import Llama, llama_backend_free
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64
import io
import gc
import os
from huggingface_hub import hf_hub_download
from JC import JC_ExtraOptions
import multiprocessing as mp
from multiprocessing.queues import Queue
import queue as _queue
import time

with open(Path(__file__).parent / "jc_data.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    CAPTION_TYPE_MAP = config["caption_type_map"]
    EXTRA_OPTIONS = config["extra_options"]
    MODEL_SETTINGS = config["model_settings"]
    CAPTION_LENGTH_CHOICES = config["caption_length_choices"]
    GGUF_MODELS = config["gguf_models"]

_MODEL_CACHE = {}
try:
    import threading as _t
    _MODEL_CACHE_LOCK = _t.RLock()
except Exception:
    _MODEL_CACHE_LOCK = None

_CHAT_HANDLER_CACHE = {}
try:
    import threading as _t
    _CHAT_HANDLER_CACHE_LOCK = _t.RLock()
except Exception:
    _CHAT_HANDLER_CACHE_LOCK = None

def _ensure_local_assets(model_name: str) -> tuple[Path, Path]:
    models_dir = Path(folder_paths.models_dir).resolve()
    llm_models_dir = (models_dir / "LLM" / "GGUF").resolve()
    llm_models_dir.mkdir(parents=True, exist_ok=True)

    model_filename = Path(model_name).name
    local_model = llm_models_dir / model_filename
    if not local_model.exists():
        if "/" not in model_name:
            raise FileNotFoundError(f"GGUF not found: {local_model}")
        repo_path, filename = model_name.rsplit("/", 1)
        local_model = Path(hf_hub_download(
            repo_id=repo_path,
            filename=filename,
            local_dir=str(llm_models_dir),
            local_dir_use_symlinks=False
        )).resolve()

    mmproj_filename = "llama-joycaption-beta-one-llava-mmproj-model-f16.gguf"
    local_mmproj = llm_models_dir / mmproj_filename
    if not local_mmproj.exists():
        local_mmproj = Path(hf_hub_download(
            repo_id="concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf",
            filename=mmproj_filename,
            local_dir=str(llm_models_dir),
            local_dir_use_symlinks=False
        )).resolve()

    return local_model.resolve(), local_mmproj.resolve()

def _get_chat_handler(mmproj_path: Path):
    key = str(mmproj_path.resolve())
    if _CHAT_HANDLER_CACHE_LOCK:
        with _CHAT_HANDLER_CACHE_LOCK:
            ch = _CHAT_HANDLER_CACHE.get(key)
            if ch is None:
                try:
                    ch = Llava15ChatHandler(clip_model_path=key, clip_n_gpu_layers=0)
                except TypeError:
                    os.environ.setdefault("LLAMA_CLIP_N_GPU_LAYERS", "0")
                    ch = Llava15ChatHandler(clip_model_path=key)
                _CHAT_HANDLER_CACHE[key] = ch
            return ch
    else:
        ch = _CHAT_HANDLER_CACHE.get(key)
        if ch is None:
            try:
                ch = Llava15ChatHandler(clip_model_path=key, clip_n_gpu_layers=0)
            except TypeError:
                os.environ.setdefault("LLAMA_CLIP_N_GPU_LAYERS", "0")
                ch = Llava15ChatHandler(clip_model_path=key)
            _CHAT_HANDLER_CACHE[key] = ch
        return ch

def _drop_all_chat_handlers():
    if _CHAT_HANDLER_CACHE_LOCK:
        with _CHAT_HANDLER_CACHE_LOCK:
            for _, _ch in list(_CHAT_HANDLER_CACHE.items()):
                for _m in ("close", "free", "__del__"):
                    try:
                        getattr(_ch, _m)()
                    except Exception:
                        pass
            _CHAT_HANDLER_CACHE.clear()
    else:
        for _, _ch in list(_CHAT_HANDLER_CACHE.items()):
            for _m in ("close", "free", "__del__"):
                try:
                    getattr(_ch, _m)()
                except Exception:
                    pass
        _CHAT_HANDLER_CACHE.clear()
    try:
        gc.collect()
    except Exception:
        pass

# -------------------------------
# Subprocess inference utilities
# -------------------------------
def _infer_worker(q: Queue, payload: dict):
    """
    Isolated process worker: initialize model, run inference once, and exit.
    Ensures CUDA/ggml contexts are destroyed with process teardown.
    """
    try:
        # 하트비트: 프로세스가 살아있음을 부모에게 즉시 알림
        try: q.put(("stage", "spawned"))
        except Exception: pass
        # 안전 장치: CLIP(mmproj)는 CPU 고정
        os.environ.setdefault("LLAMA_CLIP_N_GPU_LAYERS", "0")
        processing_mode = payload["processing_mode"]
        local_model_path = Path(payload["local_model_path"]).resolve()
        local_mmproj_path = Path(payload["local_mmproj_path"]).resolve()

        # 모델 생성
        predictor = JC_GGUF_Models(
            model="",  # unused when local paths provided
            processing_mode=processing_mode,
            local_model_path=local_model_path,
            local_mmproj_path=local_mmproj_path,
        )
        try: q.put(("stage", "model_ready"))
        except Exception: pass
        # 이미지 복원
        img_bytes = payload["image_png_bytes"]
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # 추론
        try: q.put(("stage", "gen_start"))
        except Exception: pass
        text = predictor.generate(
            image=pil_img,
            system=payload["system"],
            prompt=payload["prompt"],
            max_new_tokens=payload["max_new_tokens"],
            temperature=payload["temperature"],
            top_p=payload["top_p"],
            top_k=payload["top_k"],
        )
        try: q.put(("stage", "gen_done"))
        except Exception: pass

        q.put(("ok", text))
    except Exception as e:
        try:
            q.put(("err", f"Generation error: {e}"))
        except Exception:
            pass
    finally:
        # 예의상 최대한 비워주고 종료
        try:
            if 'predictor' in locals() and predictor is not None:
                predictor.close()
        except Exception:
            pass
        try:
            llama_backend_free()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        try: gc.collect()
        except Exception: pass

def _generate_in_subprocess(image_png_bytes: bytes,
                            model_name: str,
                            local_model_path: str,
                            local_mmproj_path: str,
                            processing_mode: str,
                            system: str,
                            prompt: str,
                            max_new_tokens: int,
                            temperature: float,
                            top_p: float,
                            top_k: int,
                            timeout_sec: int = 600) -> str:
    """
    Clear After Run 전용: 한 번의 추론을 서브프로세스에서 실행하여
    CUDA/ggml 컨텍스트를 프로세스 종료로 확실히 해제한다.
    """
    ctx = mp.get_context("spawn")
    q: Queue = ctx.Queue()
    payload = {
        "image_png_bytes": image_png_bytes,
        "model_name": model_name,
        "local_model_path": local_model_path,
        "local_mmproj_path": local_mmproj_path,
        "processing_mode": processing_mode,
        "system": system,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    p = ctx.Process(target=_infer_worker, args=(q, payload))
    p.start()
    # 워치독: 하트비트/결과 폴링 + 생존 체크
    deadline = time.time() + timeout_sec
    last_stage = "spawn_pending"
    status, data = None, None
    try:
        while time.time() < deadline:
            # 메시지 비동기 수거
            got = False
            try:
                tag, payload = q.get_nowait()
                got = True
            except _queue.Empty:
                tag, payload = None, None
            if got:
                if tag == "stage":
                    last_stage = payload or last_stage
                else:
                    status, data = tag, payload
                    break
            # 자식이 죽었는데 결과가 없다 = 크래시
            if not p.is_alive():
                ec = p.exitcode
                raise RuntimeError(f"Subprocess died at stage='{last_stage}' exitcode={ec}")
            time.sleep(0.05)
        if status is None:
            # 타임아웃
            raise RuntimeError(f"Timeout at stage='{last_stage}'")
    except Exception as e:
        try:
            p.terminate()
        except Exception:
            pass
        p.join()
        raise
    else:
        p.join()
        if status == "ok":
            return data
        elif status == "err":
            raise RuntimeError(data)
        else:
            raise RuntimeError(f"Unexpected message '{status}' at stage='{last_stage}'")

def _purge_cached_key(strong_key: str):
    """Close and remove a single cached model identified by strong_key."""
    obj = None
    if _MODEL_CACHE_LOCK:
        with _MODEL_CACHE_LOCK:
            obj = _MODEL_CACHE.pop(strong_key, None)
    else:
        obj = _MODEL_CACHE.pop(strong_key, None)
    try:
        if obj is not None:
            obj.close()
    except Exception:
        pass

def _purge_all_cached():
    """Close and remove all cached models (nuclear option)."""
    items = []
    if _MODEL_CACHE_LOCK:
        with _MODEL_CACHE_LOCK:
            items = list(_MODEL_CACHE.items())
            _MODEL_CACHE.clear()
    else:
        items = list(_MODEL_CACHE.items())
        _MODEL_CACHE.clear()
    for _, obj in items:
        try:
            obj.close()
        except Exception:
            pass

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

def build_prompt(caption_type: str, caption_length: str | int, extra_options: list[str], name_input: str) -> str:
    if caption_length == "any":
        map_idx = 0
    elif isinstance(caption_length, str) and caption_length.isdigit():
        map_idx = 1
    else:
        map_idx = 2
    
    prompt = CAPTION_TYPE_MAP[caption_type][map_idx]
    if extra_options:
        prompt += " " + " ".join(extra_options)
    return prompt.format(name=name_input or "{NAME}", length=caption_length, word_count=caption_length)

class JC_GGUF_Models:
    def __init__(self, model: str, processing_mode: str,
                 local_model_path: Path | None = None,
                 local_mmproj_path: Path | None = None):
        try:
            if local_model_path is not None and local_mmproj_path is not None:
                local_path = Path(local_model_path).resolve()
                mmproj_path = Path(local_mmproj_path).resolve()
            else:
                models_dir = Path(folder_paths.models_dir).resolve()
                llm_models_dir = (models_dir / "LLM" / "GGUF").resolve()
                llm_models_dir.mkdir(parents=True, exist_ok=True)
                model_filename = Path(model).name
                local_path = llm_models_dir / model_filename
                if not local_path.exists():
                    if "/" not in model:
                        raise ValueError("Invalid model path")
                    repo_path, filename = model.rsplit("/", 1)
                    local_path = Path(hf_hub_download(
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
            
            n_ctx = MODEL_SETTINGS["context_window"]
            n_batch = 2048
            n_threads = max(4, MODEL_SETTINGS["cpu_threads"])
            n_gpu_layers = -1 if processing_mode == "GPU" else 0
            
            self.processing_mode = processing_mode
            chat_handler = _get_chat_handler(mmproj_path)
            
            self.model = Llama(
                model_path=str(local_path),
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                chat_handler=chat_handler,
                offload_kqv=True,
                numa=(processing_mode == "CPU")
            )
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def close(self):
        try:
            if getattr(self, "model", None) is not None:
                try:
                    if hasattr(self.model, "chat_handler"):
                        self.model.chat_handler = None
                except Exception:
                    pass
                _free_llama_model(self.model)
        finally:
            self.model = None
            try:
                if getattr(self, "processing_mode", "CPU") == "GPU":
                    llama_backend_free()
            except Exception:
                pass

    def __del__(self):
        self.close()
    
    def generate(self, image: Image.Image, system: str, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> str:
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = image.resize((336, 336), Image.Resampling.BILINEAR)

            with io.BytesIO() as img_buffer:
                image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            data_uri = f"data:image/png;base64,{img_base64}"

            messages = [
                {"role": "system", "content": system.strip()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt.strip()},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                 }
            ]

            completion_params = {
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": [
                    "<|eot_id|>", "</s>",
                    "<|start_header_id|>assistant<|end_header_id|>",
                    "<|start_header_id|>user<|end_header_id|>",
                    "ASSISTANT", "ASSISTANT:", "Assistant:", "Assistant",
                    "USER:", "User:", "USER", "User",
                ],
                "stream": False,
                "repeat_penalty": 1.1,
                "mirostat_mode": 0
            }

            if top_k > 0:
                completion_params["top_k"] = top_k

            response = self.model.create_chat_completion(**completion_params)
            text = response["choices"][0]["message"]["content"] or ""

            banned_markers = (
                    "<|start_header_id|>assistant<|end_header_id|>",
                    "<|start_header_id|>user<|end_header_id|>",
                    "ASSISTANT", "ASSISTANT:", "Assistant:", "Assistant",
                    "USER:", "User:", "USER", "User",
                )
            first_hit = len(text)
            for m in banned_markers:
                pos = text.find(m)
                if pos != -1 and pos < first_hit:
                    first_hit = pos
            if first_hit != len(text):
                text = text[:first_hit].rstrip()

            for cut in ("<|eot_id|>", "</s>"):
                if cut in text:
                    text = text.split(cut, 1)[0].rstrip()

            response["choices"][0]["message"]["content"] = text
            
            del messages
            
            out = response["choices"][0]["message"]["content"].strip()

            del response
            return out
            
        except Exception as e:
            return f"Generation error: {str(e)}"

class JC_GGUF:
    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(GGUF_MODELS.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (model_list, {"default": model_list[0], "tooltip": "Select the GGUF model to use for caption generation"}),
                "processing_mode": (["GPU", "CPU"], {"default": "GPU", "tooltip": "GPU: Faster but requires more VRAM\nCPU: Slower but saves VRAM"}),
                "prompt_style": (list(CAPTION_TYPE_MAP.keys()), {"default": "Descriptive", "tooltip": "Select the style of caption you want to generate"}),
                "caption_length": (CAPTION_LENGTH_CHOICES, {"default": "any", "tooltip": "Control the length of the generated caption"}),
                "memory_management": (["Keep in Memory", "Clear After Run", "Global Cache"], {"default": "Keep in Memory", "tooltip": "Choose how to manage model memory. 'Keep in Memory' for faster processing, 'Clear After Run' for limited VRAM, 'Global Cache' for fastest processing if you have enough VRAM"}),
            },
            "optional": {
                "extra_options": ("JOYCAPTION_EXTRA_OPTIONS", {"tooltip": "Additional options to customize the caption generation"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/📝JoyCaption"

    def __init__(self):
        self.predictor = None
        self.current_processing_mode = None
        self.current_model = None
    
    def generate(self, image, model, processing_mode, prompt_style, caption_length, memory_management, extra_options=None):
        try:
            strong_key = _strong_key_from(model, processing_mode)

            if memory_management == "Clear After Run":
                pass
            elif memory_management == "Global Cache":
                try:
                    if _MODEL_CACHE_LOCK:
                        with _MODEL_CACHE_LOCK:
                            pred = _MODEL_CACHE.get(strong_key)
                            if pred is None:
                                pred = JC_GGUF_Models(GGUF_MODELS[model]["name"], processing_mode)
                                _MODEL_CACHE[strong_key] = pred
                            self.predictor = pred
                    else:
                        self.predictor = _MODEL_CACHE.get(strong_key) or JC_GGUF_Models(GGUF_MODELS[model]["name"], processing_mode)
                        _MODEL_CACHE[strong_key] = self.predictor

                    self.current_model = model
                    self.current_processing_mode = processing_mode
                except Exception as e:
                    return (f"Error loading model: {e}",)
            else:
                if (self.predictor is None or
                    self.current_model != model or
                    self.current_processing_mode != processing_mode):
                    if self.predictor is not None:
                        try:
                            self.predictor.close()
                        except Exception:
                            pass
                        self.predictor = None
                    self.predictor = JC_GGUF_Models(GGUF_MODELS[model]["name"], processing_mode)
                    self.current_model = model
                    self.current_processing_mode = processing_mode

            prompt_text = build_prompt(prompt_style, caption_length, (extra_options[0] if extra_options and len(extra_options) > 0 else []), (extra_options[1] if extra_options and len(extra_options) > 1 else "{NAME}"))

            if memory_management == "Clear After Run":
                with torch.inference_mode():
                    pil_img = ToPILImage()(image[0].permute(2, 0, 1).cpu()).convert("RGB")
                    with io.BytesIO() as buf:
                        pil_img.save(buf, format="PNG")
                        png_bytes = buf.getvalue()

                local_model_path, local_mmproj_path = _ensure_local_assets(GGUF_MODELS[model]["name"])
                response = _generate_in_subprocess(
                    image_png_bytes=png_bytes,
                    model_name=GGUF_MODELS[model]["name"],
                    local_model_path=str(local_model_path),
                    local_mmproj_path=str(local_mmproj_path),
                    processing_mode=processing_mode,
                    system=MODEL_SETTINGS["default_system_prompt"],
                    prompt=prompt_text,
                    max_new_tokens=MODEL_SETTINGS["default_max_tokens"],
                    temperature=MODEL_SETTINGS["default_temperature"],
                    top_p=MODEL_SETTINGS["default_top_p"],
                    top_k=MODEL_SETTINGS["default_top_k"],
                )
            else:
                with torch.inference_mode():
                    pil_image = ToPILImage()(image[0].permute(2, 0, 1).cpu())
                    response = self.predictor.generate(
                        image=pil_image,
                        system=MODEL_SETTINGS["default_system_prompt"],
                        prompt=prompt_text,
                        max_new_tokens=MODEL_SETTINGS["default_max_tokens"],
                        temperature=MODEL_SETTINGS["default_temperature"],
                        top_p=MODEL_SETTINGS["default_top_p"],
                        top_k=MODEL_SETTINGS["default_top_k"],
                    )

            if memory_management == "Clear After Run":
                try:
                    if self.predictor is not None:
                        self.predictor.close()
                except Exception:
                    pass
                finally:
                    self.predictor = None
                _purge_all_cached()
                _drop_all_chat_handlers()
                try:
                    llama_backend_free()
                except Exception:
                    pass
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass
                try:
                    gc.collect()
                except Exception:
                    pass

            return (response,)
        except Exception as e:
            if memory_management == "Clear After Run":
                try:
                    if getattr(self, "predictor", None) is not None:
                        self.predictor.close()
                finally:
                    self.predictor = None
                _purge_all_cached()
                _drop_all_chat_handlers()
                try:
                    llama_backend_free()
                except Exception:
                    pass
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    gc.collect()
                except Exception:
                    pass
            raise e

class JC_GGUF_adv:
    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(GGUF_MODELS.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (model_list, {"default": model_list[0], "tooltip": "Select the GGUF model to use for caption generation"}),
                "processing_mode": (["GPU", "CPU"], {"default": "GPU", "tooltip": "GPU: Faster but requires more VRAM\nCPU: Slower but saves VRAM"}),
                "prompt_style": (list(CAPTION_TYPE_MAP.keys()), {"default": "Descriptive", "tooltip": "Select the style of caption you want to generate"}),
                "caption_length": (CAPTION_LENGTH_CHOICES, {"default": "any", "tooltip": "Control the length of the generated caption"}),
                "max_new_tokens": ("INT", {"default": MODEL_SETTINGS["default_max_tokens"], "min": 1, "max": 2048, "tooltip": "Maximum number of tokens to generate. Higher values allow longer captions"}),
                "temperature": ("FLOAT", {"default": MODEL_SETTINGS["default_temperature"], "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Control the randomness of the output. Higher values make the output more creative but less predictable"}),
                "top_p": ("FLOAT", {"default": MODEL_SETTINGS["default_top_p"], "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Control the diversity of the output. Higher values allow more diverse word choices"}),
                "top_k": ("INT", {"default": MODEL_SETTINGS["default_top_k"], "min": 0, "max": 100, "tooltip": "Limit the number of possible next tokens. Lower values make the output more focused"}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Custom prompt template. If empty, will use the selected prompt style"}),
                "memory_management": (["Keep in Memory", "Clear After Run", "Global Cache"], {"default": "Keep in Memory", "tooltip": "Choose how to manage model memory. 'Keep in Memory' for faster processing, 'Clear After Run' for limited VRAM, 'Global Cache' for fastest processing if you have enough VRAM"}),
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
        self.predictor = None
        self.current_processing_mode = None
        self.current_model = None
    
    def generate(self, image, model, processing_mode, prompt_style, caption_length, max_new_tokens, temperature, top_p, top_k, custom_prompt, memory_management, extra_options=None):
        try:
            strong_key = _strong_key_from(model, processing_mode)
            
            if memory_management == "Clear After Run":
                pass
            elif memory_management == "Global Cache":
                try:
                    if _MODEL_CACHE_LOCK:
                        with _MODEL_CACHE_LOCK:
                            pred = _MODEL_CACHE.get(strong_key)
                            if pred is None:
                                pred = JC_GGUF_Models(GGUF_MODELS[model]["name"], processing_mode)
                                _MODEL_CACHE[strong_key] = pred
                            self.predictor = pred
                    else:
                        self.predictor = _MODEL_CACHE.get(strong_key) or JC_GGUF_Models(GGUF_MODELS[model]["name"], processing_mode)
                        _MODEL_CACHE[strong_key] = self.predictor
                    self.current_model = model
                    self.current_processing_mode = processing_mode
                except Exception as e:
                    return ("", f"Error loading model: {e}")
            else:
                if (self.predictor is None or
                    self.current_model != model or
                    self.current_processing_mode != processing_mode):
                    if self.predictor is not None:
                        try:
                            self.predictor.close()
                        except Exception:
                            pass
                        self.predictor = None
                    self.predictor = JC_GGUF_Models(GGUF_MODELS[model]["name"], processing_mode)
                    self.current_model = model
                    self.current_processing_mode = processing_mode

            prompt = (custom_prompt if custom_prompt.strip()
                      else build_prompt(
                          prompt_style,
                          caption_length,
                          (extra_options[0] if extra_options and len(extra_options) > 0 else []),
                          (extra_options[1] if extra_options and len(extra_options) > 1 else "{NAME}")
                      ))
            system_prompt = MODEL_SETTINGS["default_system_prompt"]
            
            if memory_management == "Clear After Run":
                with torch.inference_mode():
                    pil_img = ToPILImage()(image[0].permute(2, 0, 1).cpu()).convert("RGB")
                    with io.BytesIO() as buf:
                        pil_img.save(buf, format="PNG")
                        png_bytes = buf.getvalue()
                local_model_path, local_mmproj_path = _ensure_local_assets(GGUF_MODELS[model]["name"])
                response = _generate_in_subprocess(
                    image_png_bytes=png_bytes,
                    model_name=GGUF_MODELS[model]["name"],
                    local_model_path=str(local_model_path),
                    local_mmproj_path=str(local_mmproj_path),
                    processing_mode=processing_mode,
                    system=system_prompt,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            else:
                with torch.inference_mode():
                    pil_image = ToPILImage()(image[0].permute(2, 0, 1).cpu())
                    response = self.predictor.generate(
                        image=pil_image,
                        system=system_prompt,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )

            if memory_management == "Clear After Run":
                try:
                    if self.predictor is not None:
                        self.predictor.close()
                except Exception:
                    pass
                finally:
                    self.predictor = None
                _purge_all_cached()
                _drop_all_chat_handlers()
                try:
                    llama_backend_free()
                except Exception:
                    pass
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass
                try:
                    gc.collect()
                except Exception:
                    pass

            return (prompt, response)
        except Exception as e:
            if memory_management == "Clear After Run":
                try:
                    if getattr(self, "predictor", None) is not None:
                        self.predictor.close()
                finally:
                    self.predictor = None
                _purge_all_cached()
                _drop_all_chat_handlers()
                try:
                    llama_backend_free()
                except Exception:
                    pass
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    gc.collect()
                except Exception:
                    pass
            raise e

NODE_CLASS_MAPPINGS = {
    "JC_GGUF": JC_GGUF,
    "JC_GGUF_adv": JC_GGUF_adv,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JC_GGUF": "JoyCaption GGUF",
    "JC_GGUF_adv": "JoyCaption GGUF (Advanced)",
}








