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
        "presenting a personal project for a social network, "
        "large projection screen or digital whiteboard behind them showing clear charts, line graphs and KPIs, "
        "some charts floating slightly in 3D space like a futuristic hologram, "
        "wireless headset microphone, slim presentation clicker with laser pointer in one hand, "
        "tablet with notes on a small high table, "
        "audience in soft focus, subtle blue and violet accent lighting, "
        "no desktop computers on stage, professional yet relaxed atmosphere"
    ),
    "Профессиональное достижение": (
        "software engineer in a modern open‑space office at night, "
        "two ultra‑wide monitors with clean code editor on one screen and analytics dashboards with growing charts on the other, "
        "small notification window on screen about a successfully shipped release or passed test suite, "
        "framed award certificate or stylish tech award trophy on a nearby shelf, "
        "post‑it notes with DONE tasks on the edge of the monitor, "
        "city lights through panoramic windows, ergonomic chair, mechanical keyboard, "
        "subtle blue and purple LED light strips along the desk, "
        "feeling of shipped feature, solved hard problem and professional recognition"
    ),
    "Команда VK": (
        "diverse team of young developers and analysts in a modern social‑network office, "
        "collaborating around laptops on a large wooden table, whiteboard with wireframes, flow diagrams and product ideas, "
        "nearby coffee point with cups and snacks, cozy lounge corner with bean bags and a small library shelf in the background, "
        "glass meeting room and open coworking space instead of strict cubicles, "
        "a few team members casually discussing tasks near a high bar table, "
        "stickers with tech symbols and conference badges on backpacks, "
        "subtle blue‑violet ambient light panels and warm ceiling lights, "
        "friendly, slightly playful atmosphere of brainstorming, mentorship and shared ownership of projects"
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
