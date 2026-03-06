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


device = "cuda" if torch.cuda.is_available() else "cpu"

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
        raise FileNotFoundError(f"No checkpoint found in {models_dir}")
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
        title="Choose segmentation image",
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
        title="Save segmentation canvas",
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
WINDOW_H = 680   # Increased so the UI controls fit on screen without clipping

screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("Pix2Pix Facade Generator")

font = pygame.font.SysFont("consolas", 18)
small_font = pygame.font.SysFont("consolas", 15)

clock = pygame.time.Clock()

canvas = pygame.Surface((WIDTH, HEIGHT))

generated = None


# =========================================================
# Semantic colors (Exact CMP Facades Dataset Colors)
# =========================================================
colors = {
    "sky (1)": (223, 0, 0),         # 12 - background
    "wall (2)": (0, 0, 143),        # 1 - facade
    "window (3)": (0, 207, 255),    # 5 - window
    "door (4)": (32, 255, 223),     # 6 - door
    "roof (5)": (0, 48, 255),       # 3 - cornice
    "balcony (6)": (255, 239, 0),   # 9 - balcony
    "molding (7)": (0, 0, 223),     # 2 - molding
    "pillar (8)": (0, 128, 255),    # 4 - pillar
    "sill (9)": (112, 255, 143),    # 7 - sill
    "blind (0)": (191, 255, 64),    # 8 - blind
    "shop (-)": (255, 159, 0),      # 10 - shop
    "deco (=)": (255, 80, 0),       # 11 - deco
}

key_map = {
    pygame.K_1: "sky (1)",
    pygame.K_2: "wall (2)",
    pygame.K_3: "window (3)",
    pygame.K_4: "door (4)",
    pygame.K_5: "roof (5)",
    pygame.K_6: "balcony (6)",
    pygame.K_7: "molding (7)",
    pygame.K_8: "pillar (8)",
    pygame.K_9: "sill (9)",
    pygame.K_0: "blind (0)",
    pygame.K_MINUS: "shop (-)",
    pygame.K_EQUALS: "deco (=)",
}

color_keys = list(colors.keys())
current_color_name = "wall (2)"
current_color = colors[current_color_name]

canvas.fill(colors["sky (1)"])

# =========================================================
# Modes
# =========================================================
MODE_RECT = "rect"
MODE_FREEHAND = "freehand"
draw_mode = MODE_RECT

is_dragging_rect = False
rect_start = (0, 0)
rect_current = (0, 0)

brush_size = 10


# =========================================================
# Helpers
# =========================================================
def surface_to_pil(surface: pygame.Surface) -> Image.Image:
    arr = pygame.surfarray.array3d(surface)       # (W,H,3)
    arr = np.transpose(arr, (1, 0, 2))            # (H,W,3)
    return Image.fromarray(arr)


def pil_to_surface(img: Image.Image) -> pygame.Surface:
    img = img.convert("RGB").resize((WIDTH, HEIGHT), Image.NEAREST)
    arr = np.array(img)                           # (H,W,3)
    arr = np.transpose(arr, (1, 0, 2))            # (W,H,3)
    return pygame.surfarray.make_surface(arr)


def generate_from_surface(surface: pygame.Surface) -> pygame.Surface:
    pil = surface_to_pil(surface)
    inp = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        out = G(inp)

    out = out.squeeze(0).cpu()                    # (3,H,W)
    out = (out + 1.0) / 2.0
    out = out.clamp(0, 1)
    out = (out.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)   # (H,W,3)
    out = np.transpose(out, (1, 0, 2))                                 # (W,H,3)
    return pygame.surfarray.make_surface(out)


def clamp_to_canvas(x: int, y: int) -> tuple[int, int]:
    x = max(0, min(WIDTH - 1, x))
    y = max(0, min(HEIGHT - 1, y))
    return x, y


def make_rect(start: tuple[int, int], end: tuple[int, int]) -> pygame.Rect:
    x1, y1 = start
    x2, y2 = end
    left = min(x1, x2)
    top = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return pygame.Rect(left, top, width, height)


def draw_ui():
    panel_x = WIDTH * 2 + 12

    pygame.draw.rect(screen, (30, 30, 30), (WIDTH * 2, 0, PANEL_W, WINDOW_H))

    title = font.render("Controls & Colors", True, (255, 255, 255))
    screen.blit(title, (panel_x, 10))

    y = 40
    # Paint color controls
    for name, col in colors.items():
        pygame.draw.rect(screen, col, (panel_x, y, 22, 22))
        txt = small_font.render(name, True, (230, 230, 230))
        screen.blit(txt, (panel_x + 30, y + 2))
        y += 30

    y += 10
    lines =[
        "R = rectangle mode",
        "F = freehand mode",
        "G = generate",
        "C = clear canvas",
        "I = import seg image",
        "S = save seg canvas",
        "ESC = quit",
        "",
        f"Mode: {draw_mode}",
        f"Color: {current_color_name.split(' ')[0]}",
    ]

    for line in lines:
        surf = small_font.render(line, True, (230, 230, 230))
        screen.blit(surf, (panel_x, y))
        y += 22

    # labels
    left_label = font.render("Segmentation Input", True, (255, 255, 255))
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

            # Set current color based on defined key mapping
            elif event.key in key_map:
                current_color_name = key_map[event.key]
                current_color = colors[current_color_name]

            elif event.key == pygame.K_r:
                draw_mode = MODE_RECT
            elif event.key == pygame.K_f:
                draw_mode = MODE_FREEHAND

            elif event.key == pygame.K_c:
                canvas.fill(colors["sky (1)"])
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
                        print(f"Imported: {img_path}")
                    except Exception as e:
                        print(f"Failed to import image: {e}")

            elif event.key == pygame.K_s:
                save_path = choose_save_file()
                if save_path:
                    try:
                        surface_to_pil(canvas).save(save_path)
                        print(f"Saved canvas to: {save_path}")
                    except Exception as e:
                        print(f"Failed to save image: {e}")

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if mx < WIDTH and my < HEIGHT:
                mx, my = clamp_to_canvas(mx, my)

                if draw_mode == MODE_RECT:
                    is_dragging_rect = True
                    rect_start = (mx, my)
                    rect_current = (mx, my)

                elif draw_mode == MODE_FREEHAND:
                    pygame.draw.circle(canvas, current_color, (mx, my), brush_size)

        elif event.type == pygame.MOUSEBUTTONUP:
            mx, my = event.pos
            if draw_mode == MODE_RECT and is_dragging_rect:
                mx, my = clamp_to_canvas(mx, my)
                rect_current = (mx, my)
                rect = make_rect(rect_start, rect_current)
                if rect.width > 0 and rect.height > 0:
                    pygame.draw.rect(canvas, current_color, rect)
                is_dragging_rect = False

        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if mx < WIDTH and my < HEIGHT:
                mx, my = clamp_to_canvas(mx, my)

                if draw_mode == MODE_RECT and is_dragging_rect:
                    rect_current = (mx, my)

                elif draw_mode == MODE_FREEHAND and pygame.mouse.get_pressed()[0]:
                    pygame.draw.circle(canvas, current_color, (mx, my), brush_size)

    # draw frame
    screen.fill((15, 15, 15))

    # input canvas
    screen.blit(canvas, (0, 0))

    # preview rectangle while dragging
    if draw_mode == MODE_RECT and is_dragging_rect:
        preview = canvas.copy()
        rect = make_rect(rect_start, rect_current)
        pygame.draw.rect(preview, current_color, rect)
        screen.blit(preview, (0, 0))

        # outline
        pygame.draw.rect(screen, (255, 255, 255), rect, 2)

    # generated output
    if generated is not None:
        screen.blit(generated, (WIDTH, 0))
    else:
        blank = pygame.Surface((WIDTH, HEIGHT))
        blank.fill((25, 25, 25))
        screen.blit(blank, (WIDTH, 0))

    # borders
    pygame.draw.rect(screen, (200, 200, 200), (0, 0, WIDTH, HEIGHT), 2)
    pygame.draw.rect(screen, (200, 200, 200), (WIDTH, 0, WIDTH, HEIGHT), 2)

    draw_ui()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()