"""
🤖 Telegram Emoji Mosaic Bot
Зависимости:
    pip install python-telegram-bot Pillow opencv-python-headless numpy
Системные требования:
    ffmpeg
Переменные окружения:
    BOT_TOKEN
"""

import io
import os
import uuid
import time
import shutil
import logging
import tempfile
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputSticker
from telegram.constants import StickerFormat, StickerType
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, ConversationHandler, filters,
)

# ── Логирование ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_logs.txt", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

def log_action(user, action: str):
    name = f"{user.first_name or ''} {user.last_name or ''}".strip()
    username = f"@{user.username}" if user.username else "без username"
    log.info(f"👤 {name} ({username}, id={user.id}) | {action}")

# ── Константы ──────────────────────────────────────────────────────────────
BOT_TOKEN        = os.environ["BOT_TOKEN"]
STICKER_SIZE     = 100      # кастомные эмодзи 100×100px
MAX_STICKERS     = 120
MAX_COLS         = 13       # максимум 13 эмодзи вширь (лимит экрана телефона)
MAX_VIDEO_FRAMES = 15
MAX_VIDEO_SEC    = 3.0

# (cols, rows) — варианты сеток
GRID_OPTIONS = [
    (5,  5),
    (8,  5),
    (10, 5),
    (13, 5),
    (8,  8),
    (10, 8),
    (13, 8),
    (10, 10),
    (13, 10),
    (13, 13),
]

WAIT_FILE, WAIT_GRID, WAIT_PACK_NAME = range(3)


# ══════════════════════════════════════════════════════════════════════════
#  УТИЛИТЫ: изображение
# ══════════════════════════════════════════════════════════════════════════

def slice_image(img: Image.Image, cols: int, rows: int) -> list:
    img = img.convert("RGBA").resize(
        (cols * STICKER_SIZE, rows * STICKER_SIZE), Image.LANCZOS
    )
    cells = []
    for row in range(rows):
        for col in range(cols):
            box = (
                col * STICKER_SIZE, row * STICKER_SIZE,
                (col + 1) * STICKER_SIZE, (row + 1) * STICKER_SIZE,
            )
            cell = img.crop(box).resize((STICKER_SIZE, STICKER_SIZE), Image.LANCZOS)
            cells.append(cell)
    return cells


def ensure_valid(cell: Image.Image) -> Image.Image:
    """Гарантирует RGBA 100×100, добавляет 1 пиксель если полностью прозрачная."""
    cell = cell.convert("RGBA").resize((STICKER_SIZE, STICKER_SIZE), Image.LANCZOS)
    arr = np.array(cell)
    if arr[:, :, 3].max() == 0:
        cell.putpixel((50, 50), (255, 255, 255, 2))
    return cell


def save_cells_png(cells: list, out_dir: Path) -> list:
    paths = []
    for i, cell in enumerate(cells):
        p = out_dir / f"cell_{i:04d}.png"
        ensure_valid(cell).save(p, "PNG", optimize=True, compress_level=9)
        paths.append(p)
    return paths


# ══════════════════════════════════════════════════════════════════════════
#  УТИЛИТЫ: видео / WebP
# ══════════════════════════════════════════════════════════════════════════

def read_frames_opencv(path: Path):
    """Читает кадры через OpenCV. Возвращает (frames_BGRA, fps)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // MAX_VIDEO_FRAMES)
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            frames.append(frame)
            if len(frames) >= MAX_VIDEO_FRAMES:
                break
        idx += 1
    cap.release()
    return frames, fps


def read_frames_pillow(path: Path):
    """Читает кадры из WebP через Pillow. Возвращает (frames_BGRA, fps)."""
    img = Image.open(path)
    duration = img.info.get("duration", 100)
    fps = 1000 / duration
    frames = []
    try:
        while True:
            rgba = np.array(img.copy().convert("RGBA"))
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            frames.append(bgra)
            img.seek(img.tell() + 1)
            if len(frames) >= MAX_VIDEO_FRAMES:
                break
    except EOFError:
        pass
    return frames, fps


def split_frames_to_cells(frames: list, cols: int, rows: int) -> list:
    """Нарезает список кадров на cols×rows ячеек."""
    w = cols * STICKER_SIZE
    h = rows * STICKER_SIZE
    cells = [[] for _ in range(cols * rows)]
    for frame in frames:
        r = cv2.resize(frame, (w, h))
        for row in range(rows):
            for col in range(cols):
                y0, y1 = row * STICKER_SIZE, (row + 1) * STICKER_SIZE
                x0, x1 = col * STICKER_SIZE, (col + 1) * STICKER_SIZE
                cells[row * cols + col].append(r[y0:y1, x0:x1])
    return cells


def cell_frames_to_webm(cell_frames: list, out_path: Path, fps: float) -> Path:
    tmp = out_path.parent / f"_t_{out_path.stem}"
    tmp.mkdir(exist_ok=True)
    for i, f in enumerate(cell_frames):
        if f.shape[2] == 3:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2BGRA)
        cv2.imwrite(str(tmp / f"f_{i:04d}.png"), f)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(min(fps, 30)),
        "-i", str(tmp / "f_%04d.png"),
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-auto-alt-ref", "0",
        "-b:v", "0", "-crf", "40",
        "-t", str(MAX_VIDEO_SEC),
        "-s", f"{STICKER_SIZE}x{STICKER_SIZE}",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return out_path


# ══════════════════════════════════════════════════════════════════════════
#  TELEGRAM: создание эмодзи-пака
# ══════════════════════════════════════════════════════════════════════════

async def build_emoji_pack(ctx, user_id: int, paths: list, fmt, title: str) -> str:
    bot = ctx.bot
    me = await bot.get_me()
    uid = str(int(time.time()))[-6:]
    pack_name = f"m{uid}_{user_id}_by_{me.username}"

    all_bytes = []
    for p in paths:
        with open(p, "rb") as fh:
            all_bytes.append(fh.read())

    await bot.create_new_sticker_set(
        user_id=user_id,
        name=pack_name,
        title=title,
        stickers=[InputSticker(sticker=io.BytesIO(all_bytes[0]), emoji_list=["🟦"], format=fmt)],
        sticker_type=StickerType.CUSTOM_EMOJI,
    )
    for data in all_bytes[1:]:
        await bot.add_sticker_to_set(
            user_id=user_id,
            name=pack_name,
            sticker=InputSticker(sticker=io.BytesIO(data), emoji_list=["🟦"], format=fmt),
        )
    return pack_name


# ══════════════════════════════════════════════════════════════════════════
#  МЕНЮ
# ══════════════════════════════════════════════════════════════════════════

def main_menu_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📷 Фото-мозаика (PNG / WebP)", callback_data="mode_image")],
        [InlineKeyboardButton("🎬 Анимированная мозаика (WebM / MOV / WebP)", callback_data="mode_video")],
        [InlineKeyboardButton("📖 Инструкция", callback_data="help")],
    ])

def back_kb():
    return InlineKeyboardMarkup([[InlineKeyboardButton("← Назад", callback_data="back_menu")]])

async def show_menu(update: Update, text: str):
    kb = main_menu_kb()
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=kb)
    else:
        await update.message.reply_text(text, reply_markup=kb)


# ══════════════════════════════════════════════════════════════════════════
#  ХЭНДЛЕРЫ
# ══════════════════════════════════════════════════════════════════════════

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data.clear()
    log_action(update.effective_user, "открыл бота /start")
    await show_menu(update, "👋 Привет! Превращаю фото или видео с прозрачным фоном в эмодзи-мозаику.\n\nВыбери режим:")
    return WAIT_FILE


async def btn_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        "📖 Как пользоваться:\n\n"
        "1️⃣ Выбери режим в меню\n"
        "2️⃣ Отправь файл как документ (скрепка 📎)\n"
        "3️⃣ Выбери размер сетки\n"
        "4️⃣ Введи название эмодзи-пака\n"
        "5️⃣ Получи ссылку и добавь пак\n"
        "6️⃣ Расставь эмодзи слева→направо, сверху→вниз\n\n"
        "⚠️ Фото: PNG или статичный WebP с прозрачным фоном\n"
        "⚠️ Видео: WebM, MOV или анимированный WebP с прозрачным фоном",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("← Назад", callback_data="back_menu")]])
    )
    return WAIT_FILE


async def btn_back(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    ctx.user_data.clear()
    await show_menu(update, "Главное меню:")
    return WAIT_FILE


async def btn_mode(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    mode = q.data  # mode_image или mode_video
    ctx.user_data["mode"] = mode
    log_action(q.from_user, f"выбрал режим: {mode}")
    if mode == "mode_image":
        text = "📷 Отправь PNG или статичный WebP с прозрачным фоном как документ (скрепка 📎):"
    else:
        text = "🎬 Отправь WebM, MOV или анимированный WebP с прозрачным фоном как документ (скрепка 📎):"
    await q.edit_message_text(text, reply_markup=back_kb())
    return WAIT_FILE


async def handle_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message

    # принимаем файл из любого источника
    if msg.document:
        fobj = msg.document
        mime  = fobj.mime_type or ""
        fname = fobj.file_name or "file"
    elif msg.sticker:
        fobj  = msg.sticker
        mime  = "image/webp"
        fname = "sticker.webp"
    elif msg.animation:
        fobj  = msg.animation
        mime  = fobj.mime_type or "image/webp"
        fname = fobj.file_name or "anim.webp"
    elif msg.video:
        fobj  = msg.video
        mime  = fobj.mime_type or "video/webm"
        fname = fobj.file_name or "video.webm"
    else:
        await msg.reply_text("Отправь файл как документ (скрепка 📎).")
        return WAIT_FILE

    fl = fname.lower()
    is_png   = mime == "image/png"  or fl.endswith(".png")
    is_video = mime in ("video/webm","video/quicktime") or fl.endswith((".webm",".mov"))
    is_webp  = mime == "image/webp" or fl.endswith(".webp")

    log.info(f"Файл: mime={mime} fname={fname} png={is_png} video={is_video} webp={is_webp}")

    if not (is_png or is_video or is_webp):
        await msg.reply_text(f"❌ Формат не поддерживается ({mime}).\nНужны: PNG, WebM, MOV или WebP.")
        return WAIT_FILE

    # если режим не выбран — определяем по файлу
    if "mode" not in ctx.user_data:
        ctx.user_data["mode"] = "mode_video" if is_video else "mode_image"

    ctx.user_data["file_id"]   = fobj.file_id
    ctx.user_data["file_name"] = fname
    ctx.user_data["is_webp"]   = is_webp
    ctx.user_data["is_video"]  = is_video
    log_action(update.effective_user, f"загрузил файл: {fname}")

    grid_btns = []
    for cols, rows in GRID_OPTIONS:
        total = cols * rows
        if total > MAX_STICKERS:
            continue
        label = f"{cols}×{rows}  —  {total} эмодзи"
        if cols == MAX_COLS:
            label += "  📱 на всю ширину"
        grid_btns.append([InlineKeyboardButton(label, callback_data=f"grid_{cols}_{rows}")])
    grid_btns.append([InlineKeyboardButton("← Назад", callback_data="back_menu")])
    await msg.reply_text(
        "✅ Файл получен! Выбери размер сетки:\n"
        "📱 13 эмодзи вширь = вся ширина экрана телефона",
        reply_markup=InlineKeyboardMarkup(grid_btns)
    )
    return WAIT_GRID


async def btn_grid(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    parts = q.data.split("_")   # grid_13_8
    cols, rows = int(parts[1]), int(parts[2])
    ctx.user_data["cols"] = cols
    ctx.user_data["rows"] = rows
    total = cols * rows
    log_action(q.from_user, f"выбрал сетку {cols}×{rows}")
    note = "📱 вся ширина экрана телефона" if cols == MAX_COLS else ""
    await q.edit_message_text(
        f"📐 Сетка {cols}×{rows} ({total} эмодзи). {note}\n\nВведи название для эмодзи-пака:",
        reply_markup=back_kb()
    )
    return WAIT_PACK_NAME


async def handle_pack_name(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    title   = update.message.text.strip()[:64]
    cols    = ctx.user_data["cols"]
    rows    = ctx.user_data["rows"]
    total   = cols * rows
    file_id = ctx.user_data["file_id"]
    fname   = ctx.user_data["file_name"]
    mode    = ctx.user_data.get("mode", "mode_image")
    is_webp = ctx.user_data.get("is_webp", False)
    is_video= ctx.user_data.get("is_video", False)
    user_id = update.effective_user.id

    await update.message.reply_text(
        f"⏳ Создаю эмодзи-пак «{title}» ({cols}×{rows})...\nЭто может занять от 30 секунд до нескольких минут."
    )

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        tg_file = await ctx.bot.get_file(file_id)
        src = tmp_dir / fname
        await tg_file.download_to_drive(src)

        cells_dir = tmp_dir / "cells"
        cells_dir.mkdir()
        sticker_paths = []

        if mode == "mode_video" or is_video:
            # анимированный режим
            fmt = StickerFormat.VIDEO
            if is_webp:
                frames, fps = read_frames_pillow(src)
            else:
                frames, fps = read_frames_opencv(src)
            cell_list = split_frames_to_cells(frames, grid)
            for i, cf in enumerate(cell_list):
                out = cells_dir / f"cell_{i:04d}.webm"
                cell_frames_to_webm(cf, out, fps)
            sticker_paths = sorted(cells_dir.glob("*.webm"))
        else:
            # статичный режим
            fmt = StickerFormat.STATIC
            img = Image.open(src).convert("RGBA")
            cells = slice_image(img, grid)
            sticker_paths = save_cells_png(cells, cells_dir)

        pack_name = await build_emoji_pack(ctx, user_id, sticker_paths, fmt, title)
        log_action(update.effective_user, f"✅ создал пак «{title}» {grid}×{grid} → {pack_name}")

        await update.message.reply_text(
            f"🎉 Готово! Эмодзи-пак «{title}» создан:\n"
            f"👉 https://t.me/addemoji/{pack_name}\n\n"
            f"Как использовать:\n"
            f"1. Открой ссылку и добавь пак\n"
            f"2. Отправляй эмодзи по порядку — слева→направо, сверху→вниз ({grid} в строке)\n"
            f"3. Все {grid*grid} эмодзи вместе образуют мозаику!",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Создать ещё", callback_data="back_menu")]])
        )

    except Exception as e:
        log.exception("Ошибка при создании пака")
        await update.message.reply_text(
            f"❌ Что-то пошло не так: {e}\n\nПопробуй снова.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 В главное меню", callback_data="back_menu")]])
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    ctx.user_data.clear()
    return WAIT_FILE


# ══════════════════════════════════════════════════════════════════════════
#  ЗАПУСК
# ══════════════════════════════════════════════════════════════════════════

def main():
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .read_timeout(120)
        .write_timeout(120)
        .connect_timeout(60)
        .build()
    )

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", cmd_start)],
        states={
            WAIT_FILE: [
                CallbackQueryHandler(btn_help,  pattern="^help$"),
                CallbackQueryHandler(btn_back,  pattern="^back_menu$"),
                CallbackQueryHandler(btn_mode,  pattern="^mode_(image|video)$"),
                MessageHandler(
                    filters.Document.ALL | filters.Sticker.ALL | filters.ANIMATION | filters.VIDEO,
                    handle_file
                ),
            ],
            WAIT_GRID: [
                CallbackQueryHandler(btn_back, pattern="^back_menu$"),
                CallbackQueryHandler(btn_grid, pattern=r"^grid_\d+_\d+$"),
            ],
            WAIT_PACK_NAME: [
                CallbackQueryHandler(btn_back, pattern="^back_menu$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_pack_name),
            ],
        },
        fallbacks=[CommandHandler("start", cmd_start)],
        allow_reentry=True,
        per_message=False,
    )

    app.add_handler(conv)
    log.info("Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()
