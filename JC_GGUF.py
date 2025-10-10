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

def _get_chat_handler(mmproj_path: Path):
    key = str(mmproj_path.resolve())
    if _CHAT_HANDLER_CACHE_LOCK:
        with _CHAT_HANDLER_CACHE_LOCK:
            ch = _CHAT_HANDLER_CACHE.get(key)
            if ch is None:
                ch = Llava15ChatHandler(clip_model_path=key)
                _CHAT_HANDLER_CACHE[key] = ch
            return ch
    else:
        ch = _CHAT_HANDLER_CACHE.get(key)
        if ch is None:
            ch = Llava15ChatHandler(clip_model_path=key)
            _CHAT_HANDLER_CACHE[key] = ch
        return ch

def _drop_all_chat_handlers():
    if _CHAT_HANDLER_CACHE_LOCK:
        with _CHAT_HANDLER_CACHE_LOCK:
            _CHAT_HANDLER_CACHE.clear()
    else:
        _CHAT_HANDLER_CACHE.clear()

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
    def __init__(self, model: str, processing_mode: str):
        try:
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

            img_buffer = io.BytesIO()
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
            
            return response["choices"][0]["message"]["content"].strip()
            
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

            if memory_management == "Global Cache":
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
                _purge_all_cached()
                try:
                    if self.predictor is not None:
                        self.predictor.close()
                finally:
                    self.predictor = None
                _drop_all_chat_handlers()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass
                gc.collect()

            return (response,)
        except Exception as e:
            if memory_management == "Clear After Run":
                try:
                    if getattr(self, "predictor", None) is not None:
                        self.predictor.close()
                finally:
                    self.predictor = None
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()
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
            
            if memory_management == "Global Cache":
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
                _purge_all_cached()
                try:
                    if self.predictor is not None:
                        self.predictor.close()
                finally:
                    self.predictor = None
                _drop_all_chat_handlers()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass
                gc.collect()

            return (prompt, response)
        except Exception as e:
            if memory_management == "Clear After Run":
                try:
                    if self.predictor is not None:
                        self.predictor.close()
                finally:
                    self.predictor = None
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()
            return ("", f"Error: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "JC_GGUF": JC_GGUF,
    "JC_GGUF_adv": JC_GGUF_adv,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JC_GGUF": "JoyCaption GGUF",
    "JC_GGUF_adv": "JoyCaption GGUF (Advanced)",
}


