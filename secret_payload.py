"""Original payload encoders; no rendering/verification experiments are executed."""
import math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

def generate_random_secret_message(n_bits, device, seed):
    """Match the GPT-style setup: sample a random +/-1 secret without forced balancing."""
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    bits01 = torch.randint(0, 2, (n_bits,), generator=gen, dtype=torch.int64)
    secret = bits01 * 2 - 1
    return secret.to(device=device)

def _bytes_to_bits(payload):
    bits = []
    for byte in payload:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits

def _closest_factor_pair(n):
    root = int(math.sqrt(n))
    for h in range(root, 0, -1):
        if n % h == 0:
            return n // h, h
    return n, 1

def _resolve_secret_image_shape(n_bits, width, height):
    if width > 0 and height > 0:
        if width * height != n_bits:
            raise ValueError(f"secret image shape {width}x{height} does not match n_bits={n_bits}")
        return width, height
    if width > 0:
        if n_bits % width != 0:
            raise ValueError(f"n_bits={n_bits} is not divisible by secret_image_width={width}")
        return width, n_bits // width
    if height > 0:
        if n_bits % height != 0:
            raise ValueError(f"n_bits={n_bits} is not divisible by secret_image_height={height}")
        return n_bits // height, height
    return _closest_factor_pair(n_bits)

def _load_secret_font(font_path, font_size):
    """Let Pillow locate installed fonts; allow an explicit font file override."""
    font_candidates = []
    if font_path:
        font_candidates.append(font_path)
    font_candidates.extend([
        "NotoSansCJK-Regular.ttc",
        "NotoSerifCJK-Regular.ttc",
        "wqy-zenhei.ttc",
        "wqy-microhei.ttc",
        "ukai.ttc",
        "uming.ttc",
        "SourceHanSansSC-Regular.otf",
        "SourceHanSerifSC-Regular.otf",
        "DejaVuSans.ttf",
    ])
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()

def _contains_non_ascii(text):
    return any(ord(ch) > 127 for ch in text)

def encode_text_secret_message(secret_text, n_bits, device, seed):
    payload = secret_text.encode('utf-8')
    payload_len = len(payload)
    header_bits = [(payload_len >> shift) & 1 for shift in range(31, -1, -1)]
    payload_bits = _bytes_to_bits(payload)
    total_bits = len(header_bits) + len(payload_bits)
    if total_bits > n_bits:
        raise ValueError(
            f"secret_text requires {total_bits} bits but n_bits={n_bits}. "
            "Increase --n_deltas or shorten --secret_text."
        )
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    filler_len = n_bits - total_bits
    filler = torch.randint(0, 2, (filler_len,), generator=gen, dtype=torch.int64).tolist()
    bits01 = torch.tensor(header_bits + payload_bits + filler, dtype=torch.int64)
    return bits01.mul(2).sub(1).to(device=device)

def _draw_nn232_logo(image):
    width, height = image.size
    draw = ImageDraw.Draw(image)
    margin_x = max(2, width // 8)
    margin_y = max(2, height // 8)
    xs = [
        margin_x,
        width // 2,
        width - 1 - margin_x,
    ]
    top_y = margin_y
    bottom_y = height - 1 - margin_y
    mid_y = height // 2
    nodes = {
        'L1': (xs[0], top_y),
        'L2': (xs[0], bottom_y),
        'H1': (xs[1], top_y),
        'H2': (xs[1], mid_y),
        'H3': (xs[1], bottom_y),
        'R1': (xs[2], top_y),
        'R2': (xs[2], bottom_y),
    }
    radius = max(1, min(width, height) // 10)
    line_width = max(1, radius // 2)
    edges = [
        ('L1', 'H1'),
        ('L1', 'H2'),
        ('L2', 'H2'),
        ('L2', 'H3'),
        ('H1', 'R1'),
        ('H2', 'R1'),
        ('H2', 'R2'),
        ('H3', 'R2'),
    ]
    for src, dst in edges:
        draw.line([nodes[src], nodes[dst]], fill=0, width=line_width)
    for x, y in nodes.values():
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=0)

def _draw_bird_logo(image):
    width, height = image.size
    draw = ImageDraw.Draw(image)

    def sx(x):
        return int(round(x * (width - 1) / 31.0))

    def sy(y):
        return int(round(y * (height - 1) / 31.0))

    def poly(points, fill):
        draw.polygon([(sx(x), sy(y)) for x, y in points], fill=fill)

    left_wing = [(3, 18), (7, 13), (10, 9), (14, 4), (18, 2), (20, 5), (18, 9), (15, 13), (11, 17), (7, 19)]
    body = [(8, 18), (13, 15), (18, 14), (22, 15), (25, 17), (26, 20), (24, 23), (19, 25), (13, 25), (9, 23)]
    tail = [(9, 21), (5, 26), (10, 24), (12, 22)]
    neck = [(16, 14), (20, 11), (24, 10), (27, 11), (29, 13), (28, 15), (25, 16), (21, 16)]
    beak = [(26, 11), (31, 12), (26, 14)]

    poly(left_wing, fill=0)
    poly(body, fill=0)
    poly(tail, fill=0)
    poly(neck, fill=0)
    draw.ellipse((sx(20), sy(9), sx(26), sy(15)), fill=0)
    poly(beak, fill=0)

    wing_cut = [(9, 17), (12, 13), (15, 9), (16, 11), (14, 15), (11, 18)]
    poly(wing_cut, fill=255)
    draw.point((sx(24), sy(11)), fill=255)

def _draw_lock_s_logo(image):
    base = Image.new('L', (32, 32), color=255)
    draw = ImageDraw.Draw(base)

    rows = [
        "." * 32,
        "." * 32,
        "." * 32,
        "." * 32,
        "." * 32,
        "." * 13 + "#" * 6 + "." * 13,
        "." * 11 + "#" * 10 + "." * 11,
        "." * 10 + "#" * 12 + "." * 10,
        "." * 8 + "#" * 4 + "." * 8 + "#" * 4 + "." * 8,
        "." * 8 + "#" * 3 + "." * 10 + "#" * 3 + "." * 8,
        "." * 8 + "#" * 3 + "." * 10 + "#" * 3 + "." * 8,
        "." * 8 + "#" * 3 + "." * 10 + "#" * 3 + "." * 8,
        "." * 7 + "#" * 4 + "." * 10 + "#" * 4 + "." * 7,
        "." * 7 + "#" * 18 + "." * 7,
        "." * 7 + "#" * 18 + "." * 7,
        "." * 6 + "#" * 20 + "." * 6,
        "." * 6 + "#" * 20 + "." * 6,
        "." * 6 + "#" * 20 + "." * 6,
        "." * 6 + "#" * 20 + "." * 6,
        "." * 6 + "#" * 20 + "." * 6,
        "." * 6 + "#" * 20 + "." * 6,
        "." * 6 + "#" * 20 + "." * 6,
        "." * 6 + "#" * 20 + "." * 6,
        "." * 6 + "#" * 20 + "." * 6,
        "." * 6 + "#" * 20 + "." * 6,
        "." * 6 + "#" * 20 + "." * 6,
        "." * 7 + "#" * 18 + "." * 7,
        "." * 8 + "#" * 16 + "." * 8,
        "." * 9 + "#" * 14 + "." * 9,
        "." * 32,
        "." * 32,
        "." * 32,
    ]
    pixels = base.load()
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#":
                pixels[x, y] = 0

    draw.rounded_rectangle((11, 19, 20, 26), radius=1, fill=255)

    draw.rectangle((14, 20, 17, 20), fill=0)
    draw.rectangle((14, 21, 14, 22), fill=0)
    draw.rectangle((14, 23, 17, 23), fill=0)
    draw.rectangle((17, 24, 17, 25), fill=0)
    draw.rectangle((14, 26, 17, 26), fill=0)

    resized = base.resize(image.size, Image.Resampling.NEAREST)
    image.paste(resized)

def encode_binary_text_image_secret(secret_text, n_bits, device, width=0, height=0, font_path='', logo_style='none'):
    width, height = _resolve_secret_image_shape(n_bits, width, height)
    if logo_style == 'nn232':
        image = Image.new('L', (width, height), color=255)
        _draw_nn232_logo(image)
    elif logo_style in ('bird', 'bird_s'):
        image = Image.new('L', (width, height), color=255)
        _draw_bird_logo(image)
    elif logo_style == 'lock_s':
        image = Image.new('L', (width, height), color=255)
        _draw_lock_s_logo(image)
    else:
        if not secret_text:
            raise ValueError(
                "binary_text_image mode needs --secret_text unless --secret_logo_style is used."
            )
        if _contains_non_ascii(secret_text):
            probe_font = _load_secret_font(font_path, max(8, min(width, height)))
            if getattr(probe_font, 'path', None) is None and font_path == '':
                raise ValueError(
                    "binary_text_image mode needs a CJK-capable font for non-ASCII text. "
                    "Please pass --secret_font_path to a Chinese-capable font file."
                )
        image = Image.new('L', (width, height), color=255)
        draw = ImageDraw.Draw(image)
        lines = secret_text.splitlines() or ['']

        placed = False
        max_font_size = max(8, min(width, height))
        for font_size in range(max_font_size, 3, -1):
            font = _load_secret_font(font_path, font_size)
            line_boxes = [draw.textbbox((0, 0), line or ' ', font=font) for line in lines]
            line_widths = [box[2] - box[0] for box in line_boxes]
            line_heights = [max(1, box[3] - box[1]) for box in line_boxes]
            text_width = max(line_widths) if line_widths else 0
            text_height = sum(line_heights)
            if len(lines) > 1:
                text_height += max(1, font_size // 6) * (len(lines) - 1)
            if text_width <= max(1, width - 2) and text_height <= max(1, height - 2):
                y = max(0, (height - text_height) // 2)
                gap = max(1, font_size // 6)
                for idx, line in enumerate(lines):
                    box = line_boxes[idx]
                    w = box[2] - box[0]
                    h = max(1, box[3] - box[1])
                    x = max(0, (width - w) // 2)
                    draw.text((x, y - box[1]), line, fill=0, font=font)
                    y += h + gap
                placed = True
                break
        if not placed:
            raise ValueError(
                f"Unable to fit secret_text into {width}x{height} binary image. "
                "Increase --n_deltas / adjust secret_image_width,height / shorten text."
            )

    bits01 = np.asarray(image, dtype=np.uint8)
    bits01 = (bits01 < 128).astype(np.int64).reshape(-1)
    return torch.from_numpy(bits01).mul(2).sub(1).to(device=device), {'width': width, 'height': height}

def _secret_pm_to_bits01(secret_bits_pm):
    return [1 if int(bit) > 0 else 0 for bit in secret_bits_pm]

def _secret_mask(n_bits, seed):
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(seed))
    return torch.randint(0, 2, (n_bits,), generator=gen, dtype=torch.int64)

def whiten_secret(secret_pm, seed):
    """XOR whitening only; preserve all response-position indices."""
    secret_cpu = secret_pm.detach().cpu().to(dtype=torch.int64)
    bits01 = (secret_cpu > 0).to(dtype=torch.int64)
    mask = _secret_mask(bits01.numel(), seed)
    embedded_bits = torch.bitwise_xor(bits01, mask)
    return embedded_bits.mul(2).sub(1).to(device=secret_pm.device)

def restore_whitened_secret_bits(secret_bits_pm, metadata):
    """Undo XOR whitening on the complete registered bit vector."""
    bits01 = _secret_pm_to_bits01(secret_bits_pm)
    transform = metadata.get('bit_transform') if isinstance(metadata, dict) else None
    if not transform:
        return [1 if bit > 0 else -1 for bit in secret_bits_pm]
    if transform.get('type') != 'xor':
        raise ValueError('Expected XOR-only metadata; older permuted artifacts require their original decoder')

    n_bits = int(transform.get('n_bits', len(bits01)))
    if len(bits01) != n_bits:
        raise ValueError(f'Inverse whitening needs {n_bits} bits, received {len(bits01)}')

    mask = _secret_mask(n_bits, int(transform['seed']))
    embedded = torch.tensor(bits01, dtype=torch.int64)
    raw_bits = torch.bitwise_xor(embedded, mask)
    return raw_bits.mul(2).sub(1).tolist()

def add_secret_transform_metadata(metadata, n_bits, seed):
    metadata = dict(metadata)
    metadata['bit_transform'] = {
        'type': 'xor',
        'seed': int(seed),
        'n_bits': int(n_bits),
    }
    metadata['embedded_secret_stage'] = 'post_whitening'
    metadata['raw_secret_stage'] = 'pre_whitening'
    metadata['whitening_key_visibility'] = 'owner_private'
    metadata['position_permutation'] = False
    return metadata
