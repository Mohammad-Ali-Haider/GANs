import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pygame
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# =========================================================
# Device
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Generator (must match your training architecture)
# =========================================================
def down_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, True)
    )

def up_block(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(True)
    )

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = down_block(3, 64)
        self.d2 = down_block(64, 128)
        self.d3 = down_block(128, 256)
        self.d4 = down_block(256, 512)
        self.d5 = down_block(512, 512)

        self.u1 = up_block(512, 512)
        self.u2 = up_block(1024, 256)
        self.u3 = up_block(512, 128)
        self.u4 = up_block(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)

        u1 = self.u1(d5)
        u2 = self.u2(torch.cat([u1, d4], dim=1))
        u3 = self.u3(torch.cat([u2, d3], dim=1))
        u4 = self.u4(torch.cat([u3, d2], dim=1))

        return self.final(torch.cat([u4, d1], dim=1))


# =========================================================
# Load latest model
# =========================================================
def find_latest_checkpoint(models_dir: Path) -> Path:
    ckpts = sorted(models_dir.glob("ckpt_epoch_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {models_dir}. Did you train the model?")
    return ckpts[-1]

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

ckpt_path = find_latest_checkpoint(MODELS_DIR)
print(f"Loading model: {ckpt_path}")

G = Generator().to(device)
checkpoint = torch.load(ckpt_path, map_location=device)
G.load_state_dict(checkpoint["G"])
G.eval()

# =========================================================
# Transform
# =========================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# =========================================================
# File dialog helpers
# =========================================================
def choose_image_file() -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Choose sketch image",
        filetypes=[
            ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path if path else None


def choose_save_file() -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.asksaveasfilename(
        title="Save generated canvas",
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")],
    )
    root.destroy()
    return path if path else None


# =========================================================
# Pygame setup
# =========================================================
pygame.init()
pygame.font.init()

WIDTH = 256
HEIGHT = 256
PANEL_W = 320
WINDOW_W = WIDTH * 2 + PANEL_W
WINDOW_H = HEIGHT + 140

screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("Pix2Pix Edge-to-Image Generator")

font = pygame.font.SysFont("consolas", 18)
small_font = pygame.font.SysFont("consolas", 15)

clock = pygame.time.Clock()

canvas = pygame.Surface((WIDTH, HEIGHT))
# Start with a white background for edge-to-image
canvas.fill((255, 255, 255))

generated = None


# =========================================================
# Drawing State
# =========================================================
colors = {
    "Pen (Black)": (0, 0, 0),
    "Eraser (White)": (255, 255, 255),
}
current_tool = "Pen (Black)"
current_color = colors[current_tool]

is_drawing = False
last_pos = None
brush_size = 2   # Thinner brush default for edge drawings


# =========================================================
# Helpers
# =========================================================
def surface_to_pil(surface: pygame.Surface) -> Image.Image:
    arr = pygame.surfarray.array3d(surface)
    arr = np.transpose(arr, (1, 0, 2))
    return Image.fromarray(arr)


def pil_to_surface(img: Image.Image) -> pygame.Surface:
    img = img.convert("RGB").resize((WIDTH, HEIGHT), Image.NEAREST)
    arr = np.array(img)
    arr = np.transpose(arr, (1, 0, 2))
    return pygame.surfarray.make_surface(arr)


def generate_from_surface(surface: pygame.Surface) -> pygame.Surface:
    pil = surface_to_pil(surface)
    inp = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        out = G(inp)

    out = out.squeeze(0).cpu()
    out = (out + 1.0) / 2.0
    out = out.clamp(0, 1)
    out = (out.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    out = np.transpose(out, (1, 0, 2))
    return pygame.surfarray.make_surface(out)


def clamp_to_canvas(x: int, y: int) -> tuple[int, int]:
    return max(0, min(WIDTH - 1, x)), max(0, min(HEIGHT - 1, y))


def draw_ui():
    panel_x = WIDTH * 2 + 12

    pygame.draw.rect(screen, (30, 30, 30), (WIDTH * 2, 0, PANEL_W, WINDOW_H))

    title = font.render("Tools", True, (255, 255, 255))
    screen.blit(title, (panel_x, 10))

    lines =[
        "1 = Pen (Black)",
        "2 = Eraser (White)",
        "[ = Decrease Brush Size",
        "] = Increase Brush Size",
        "",
        "G = Generate Image",
        "C = Clear Canvas",
        "I = Import Sketch",
        "S = Save Output",
        "ESC = Quit",
        "",
        f"Active Tool: {current_tool}",
        f"Brush Size: {brush_size} px",
    ]

    y = 40
    for line in lines:
        surf = small_font.render(line, True, (230, 230, 230))
        screen.blit(surf, (panel_x, y))
        y += 22

    left_label = font.render("Your Sketch Input", True, (255, 255, 255))
    right_label = font.render("Generated Output", True, (255, 255, 255))
    screen.blit(left_label, (20, HEIGHT + 10))
    screen.blit(right_label, (WIDTH + 20, HEIGHT + 10))


# =========================================================
# Main loop
# =========================================================
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_1:
                current_tool = "Pen (Black)"
                current_color = colors[current_tool]
            elif event.key == pygame.K_2:
                current_tool = "Eraser (White)"
                current_color = colors[current_tool]
            elif event.key == pygame.K_LEFTBRACKET:
                brush_size = max(1, brush_size - 1)
            elif event.key == pygame.K_RIGHTBRACKET:
                brush_size = min(30, brush_size + 1)

            elif event.key == pygame.K_c:
                canvas.fill((255, 255, 255))
                generated = None
            elif event.key == pygame.K_g:
                generated = generate_from_surface(canvas)
            elif event.key == pygame.K_i:
                img_path = choose_image_file()
                if img_path:
                    try:
                        imported = Image.open(img_path).convert("RGB").resize((WIDTH, HEIGHT), Image.NEAREST)
                        canvas = pil_to_surface(imported)
                        generated = None
                    except Exception as e:
                        print(f"Failed to import: {e}")
            elif event.key == pygame.K_s:
                save_path = choose_save_file()
                if save_path:
                    try:
                        if generated is not None:
                            surface_to_pil(generated).save(save_path)
                            print(f"Saved generated image to: {save_path}")
                        else:
                            surface_to_pil(canvas).save(save_path)
                            print(f"Saved sketch to: {save_path}")
                    except Exception as e:
                        print(f"Failed to save: {e}")

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if mx < WIDTH and my < HEIGHT:
                mx, my = clamp_to_canvas(mx, my)
                is_drawing = True
                last_pos = (mx, my)
                # Draw initial dot
                pygame.draw.circle(canvas, current_color, (mx, my), brush_size // 2)

        elif event.type == pygame.MOUSEBUTTONUP:
            is_drawing = False
            last_pos = None

        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if is_drawing and mx < WIDTH and my < HEIGHT:
                mx, my = clamp_to_canvas(mx, my)
                # Draw a line from the last mouse position to current to avoid dotted gaps
                if last_pos is not None:
                    pygame.draw.line(canvas, current_color, last_pos, (mx, my), brush_size)
                    pygame.draw.circle(canvas, current_color, (mx, my), brush_size // 2)
                last_pos = (mx, my)

    screen.fill((15, 15, 15))
    screen.blit(canvas, (0, 0))

    if generated is not None:
        screen.blit(generated, (WIDTH, 0))
    else:
        blank = pygame.Surface((WIDTH, HEIGHT))
        blank.fill((25, 25, 25))
        screen.blit(blank, (WIDTH, 0))

    pygame.draw.rect(screen, (200, 200, 200), (0, 0, WIDTH, HEIGHT), 2)
    pygame.draw.rect(screen, (200, 200, 200), (WIDTH, 0, WIDTH, HEIGHT), 2)

    draw_ui()

    pygame.display.flip()
    clock.tick(120)

pygame.quit()
sys.exit()