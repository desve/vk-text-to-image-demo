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
        "young software developer on a small stage in a modern tech hub, "
        "presenting a personal project for a social network, large projection screen with clean UI mockups, "
        "subtle blue and violet accent lighting, audience of young people with laptops, "
        "sleek minimalistic interior, glass panels, cable‑managed equipment, "
        "professional yet relaxed atmosphere"
    ),
    "Профессиональное достижение": (
        "software engineer in a modern open‑space office at night, "
        "two ultra‑wide monitors with code editor and analytics dashboards, pinned sticky notes with TODOs, "
        "city lights through panoramic windows, ergonomic chair, mechanical keyboard, noise‑canceling headphones on the desk, "
        "soft blue and purple ambient light strips along the wall, "
        "focused and inspired mood, minimalistic but detail‑rich workspace"
    ),
    "Команда VK": (
        "diverse team of young developers and analysts in a modern social‑network office, "
        "collaborating around laptops on a large wooden table, whiteboard with wireframes and flow diagrams, "
        "glass meeting room, cozy corner with bean bags and floor lamp in the background, "
        "stickers with tech symbols on laptops, subtle blue‑violet brand‑like light panels, "
        "friendly and creative atmosphere, clean minimal design with many small office details"
    ),
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
    negative_prompt: str | None = None,
):
    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
    )
    return result.images[0]

DEFAULT_NEGATIVE_PROMPT = (
    "animals, cat, dog, cartoon animal, low quality, blurry, distorted face, "
    "text, watermark, logo, signature"
)


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
    negative_prompt: str | None = None,
):
    base_prompt = build_scene_prompt(scene_name, user_text)
    full_prompt = build_prompt(base_prompt, style_key)
    neg = negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT

    img = generate_one(
        full_prompt,
        steps=steps,
        guidance=guidance,
        height=height,
        width=width,
        negative_prompt=neg,
    )
    return img
