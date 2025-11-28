import torch
from diffusers import StableDiffusionXLPipeline

# ---- инициализация пайплайна ----
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)
pipe.enable_attention_slicing()

# ---- базовые промпты сцен ----
base_prompts = {
    "Самопрезентация": (
        "one young AI engineer on stage with microphone, "
        "big screen behind showing neural network diagram, tech conference"
    ),
    "Профессиональное достижение": (
        "data scientist working late at night in a modern office, "
        "only monitor light and city lights outside window, charts and neural networks on screens"
    ),
    "Команда VK": (
        "three interns in a modern IT office, sitting around a round table with laptops and sticky notes, "
        "warm friendly atmosphere"
    )
}

# ---- стили ----
styles = {
    "vk_flat": "flat illustration, clean lines, blue and violet colors",
    "neon": "neon cyberpunk style, glowing edges, high contrast"
}

def build_prompt(base: str, style_key: str) -> str:
    style_tail = styles.get(style_key, "")
    if style_tail:
        return f"{base}, {style_tail}"
    return base

def generate_one(
    prompt: str,
    steps: int = 25,
    guidance: float = 7.5,
    height: int = 512,
    width: int = 512,
):
    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
    )
    return result.images[0]

def translate_to_english(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    if not text:
        return ""
    return text

def build_scene_prompt(scene_name: str, user_text: str | None = None) -> str:
    user_text = (user_text or "").strip()
    user_text_en = translate_to_english(user_text) if user_text else ""

    if scene_name == "Свободный текст":
        if user_text_en:
            return user_text_en
        return "abstract illustration of career, technology and collaboration"

    base_en = base_prompts.get(scene_name, "")
    if user_text_en:
        return f"{base_en}, {user_text_en}"
    return base_en

def generate_image_for_app(
    scene_name: str,
    style_key: str,
    user_text: str,
    steps: int = 25,
    guidance: float = 7.5,
    height: int = 512,
    width: int = 512,
):
    base_prompt = build_scene_prompt(scene_name, user_text)
    full_prompt = build_prompt(base_prompt, style_key)
    img = generate_one(
        full_prompt,
        steps=steps,
        guidance=guidance,
        height=height,
        width=width,
    )
    return img
