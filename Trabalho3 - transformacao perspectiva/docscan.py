#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}
PAPERS = {"a4": 210 / 297, "a5": 148 / 210, "letter": 8.5 / 11, "legal": 8.5 / 14}
DEFAULT_FOCAL_FRAC = 0.8

def order_points(pts) -> np.ndarray:
    """Ordena 4 pontos como TL, TR, BR, BL.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    cw = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    start = int(np.argmin(cw.sum(axis=1)))
    return np.roll(cw, -start, axis=0).astype(np.float32)


def parse_corners(text: str) -> np.ndarray:
    pts = [tuple(float(v) for v in p.split(",")) for p in text.split(";") if p.strip()]
    if len(pts) != 4 or any(len(p) != 2 for p in pts):
        raise ValueError('--corners deve ter 4 pontos: "x1,y1;x2,y2;x3,y3;x4,y4"')
    return order_points(np.array(pts, dtype=np.float32))


def side_lengths(q: np.ndarray) -> tuple[float, float]:
    tl, tr, br, bl = q
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    return float(w), float(h)

def estimate_aspect_ratio(q: np.ndarray, image_shape, focal: float | None = None) -> float | None:
    h_img, w_img = image_shape[:2]
    u0, v0 = w_img / 2.0, h_img / 2.0
    tl, tr, br, bl = q
    m1, m2, m3, m4 = (np.array([p[0], p[1], 1.0]) for p in (tl, tr, bl, br))
    d2 = np.dot(np.cross(m2, m4), m3)
    d3 = np.dot(np.cross(m3, m4), m2)
    if abs(d2) < 1e-9 or abs(d3) < 1e-9:
        return None
    k2 = np.dot(np.cross(m1, m4), m3) / d2
    k3 = np.dot(np.cross(m1, m4), m2) / d3
    n2, n3 = k2 * m2 - m1, k3 * m3 - m1
    if focal is not None:
        f = float(focal)
    else:
        if abs(n2[2]) < 1e-6 or abs(n3[2]) < 1e-6:
            return None
        f2 = -1.0 / (n2[2] * n3[2]) * (
            (n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0**2)
            + (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0**2)
        )
        if not np.isfinite(f2) or f2 <= 0:
            return None
        f = np.sqrt(f2)
    Ai = np.linalg.inv(np.array([[f, 0, u0], [0, f, v0], [0, 0, 1.0]]))
    M = Ai.T @ Ai
    num, den = n2 @ M @ n2, n3 @ M @ n3
    if num <= 0 or den <= 0:
        return None
    return float(np.sqrt(num / den))


def auto_aspect(q: np.ndarray, image_shape, focal: float | None = None) -> float:
    """Cadeia: focal informada > autocalibração > focal típica > razão dos lados."""
    return (
        estimate_aspect_ratio(q, image_shape, focal)
        or estimate_aspect_ratio(q, image_shape)
        or estimate_aspect_ratio(q, image_shape, DEFAULT_FOCAL_FRAC * max(image_shape[:2]))
        or (lambda wh: wh[0] / wh[1])(side_lengths(q))
    )


def parse_aspect(text: str | None) -> float | None:
    """'a4', '1.414', '210:297' ou '210/297' -> largura/altura. None/'auto' -> None."""
    if text is None or text.lower() == "auto":
        return None
    if text.lower() in PAPERS:
        return PAPERS[text.lower()]
    for sep in (":", "/"):
        if sep in text:
            a, b = text.split(sep)
            return float(a) / float(b)
    return float(text)


def snap_orientation(aspect: float, estimated: float) -> float:
    return aspect if abs(np.log(aspect / estimated)) <= abs(np.log(1 / aspect / estimated)) else 1 / aspect

def rectify(
    img: np.ndarray,
    quad: np.ndarray,
    aspect: float | None = None,
    out_width: int | None = None,
    interp: str = "cubic",
    focal: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Leva o quadrilátero `quad` (TL,TR,BR,BL) a um retângulo frontal. Retorna (imagem, H)."""
    quad = order_points(quad)
    w_px, h_px = side_lengths(quad)
    if aspect is None:
        aspect = auto_aspect(quad, img.shape, focal)
    W = int(out_width or round(max(w_px, h_px * aspect)))
    H_ = int(round(W / aspect))
    dst = np.array([[-0.5, -0.5], [W - 0.5, -0.5], [W - 0.5, H_ - 0.5], [-0.5, H_ - 0.5]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, H, (W, H_), flags=INTERP[interp]), H


def trim_border(img: np.ndarray, frac: float = 0.005) -> np.ndarray:
    """Apara uma faixa fina (fração da largura) em cada lado: remove pixels de borda misturados com a mesa."""
    t = int(round(frac * img.shape[1]))
    return img[t: img.shape[0] - t, t: img.shape[1] - t] if t > 0 else img


def _chroma(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)


def _edge_map(gray: np.ndarray, chroma: np.ndarray | None) -> np.ndarray:
    e = cv2.Canny(gray, 25, 75)
    if chroma is not None:
        c8 = cv2.GaussianBlur(np.clip(chroma * 6, 0, 255).astype(np.uint8), (5, 5), 0)
        e = cv2.bitwise_or(e, cv2.Canny(c8, 20, 60))
    return cv2.dilate(e, np.ones((5, 5), np.uint8))


def _edge_support(quad: np.ndarray, edges: np.ndarray, n: int = 240) -> float:
    pts = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        t = np.linspace(0, 1, n // 4, endpoint=False)[:, None]
        pts.append(a + t * (b - a))
    pts = np.round(np.vstack(pts)).astype(int)
    h, w = edges.shape
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    hit = np.zeros(len(pts), bool)
    hit[ok] = edges[pts[ok, 1], pts[ok, 0]] > 0
    return float(hit.mean())


def _quads_from_mask(mask: np.ndarray, min_area: float, frame_hw) -> list[np.ndarray]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    fh, fw = frame_hw
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:6]:
        if cv2.contourArea(c) < min_area:
            break
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        for eps in (0.02, 0.03, 0.05):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                x, y, bw, bh = cv2.boundingRect(ap)
                if bw >= 0.98 * fw and bh >= 0.98 * fh:
                    break
                out.append(ap.reshape(4, 2).astype(np.float32))
                break
    return out


def detect_document(
    img: np.ndarray,
    min_area_frac: float = 0.15,
    chroma: bool = True,
    score: bool = True,
    refine: bool = True,
    return_info: bool = False,
):
    s = min(1.0, 800.0 / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    chrom = _chroma(cv2.GaussianBlur(small, (7, 7), 0)) if chroma else None
    k = np.ones((3, 3), np.uint8)

    masks = []
    edges = cv2.dilate(cv2.Canny(gray, 40, 120), k, iterations=2)
    cn, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(edges)
    cv2.drawContours(filled, cn, -1, 255, cv2.FILLED)
    masks.append(cv2.erode(filled, k, iterations=2))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)))
    if chroma:
        c8 = np.clip(chrom * 8, 0, 255).astype(np.uint8)
        t, cm = cv2.threshold(c8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masks.append(cv2.morphologyEx(cm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)))
        masks[-1] = cv2.morphologyEx(masks[-1], cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    min_area = min_area_frac * small.shape[0] * small.shape[1]
    cands = []
    for m in masks:
        cands += _quads_from_mask(m, min_area, small.shape[:2])
    if not cands:
        return (None, {}) if return_info else None

    emap = _edge_map(gray, chrom)
    area_total = small.shape[0] * small.shape[1]
    best, best_val, best_sup = None, -1.0, 0.0
    for q in cands:
        area = cv2.contourArea(q.reshape(-1, 1, 2)) / area_total
        sup = _edge_support(order_points(q), emap)
        val = np.sqrt(area) * sup if score else area
        if score and sup < 0.35:
            continue
        if val > best_val:
            best, best_val, best_sup = q, val, sup
    if best is None:
        return (None, {}) if return_info else None
    quad = order_points(best / s)
    if refine:
        quad = refine_corners(img, quad)
    return (quad, {"support": best_sup}) if return_info else quad


def refine_corners(img: np.ndarray, quad: np.ndarray, n_samples: int = 60, search_frac: float = 0.008) -> np.ndarray:
    quad = order_points(quad)
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.2).astype(np.float32)
    cr = _chroma(cv2.GaussianBlur(img, (0, 0), 1.2))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) + 1.5 * cv2.Sobel(cr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) + 1.5 * cv2.Sobel(cr, cv2.CV_32F, 0, 1, ksize=3)
    r = max(4.0, search_frac * max(img.shape[:2]))
    offs = np.arange(-r, r + 0.01, 0.5, dtype=np.float32)
    centroid = quad.mean(axis=0)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        L = float(np.linalg.norm(d))
        t = d / L
        nrm = np.array([t[1], -t[0]], np.float32)
        if np.dot(nrm, (a + b) / 2 - centroid) < 0:
            nrm = -nrm
        s_ = np.linspace(0.12, 0.88, n_samples, dtype=np.float32)
        cx = (a[0] + s_ * d[0])[:, None] + offs[None, :] * nrm[0]
        cy = (a[1] + s_ * d[1])[:, None] + offs[None, :] * nrm[1]
        prof = cv2.remap(gx, cx, cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) * nrm[0] \
             + cv2.remap(gy, cx, cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) * nrm[1]
        k = np.argmax(np.abs(prof), axis=1)
        peak = np.abs(prof[np.arange(len(k)), k])
        keep = peak > max(0.35 * np.median(peak), 1e-3)
        pts = np.stack([cx[np.arange(len(k)), k], cy[np.arange(len(k)), k]], axis=1)[keep]
        if len(pts) < n_samples // 2:
            lines.append((a, t))
            continue
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float32), np.array([vx, vy], np.float32)))
    out = []
    for i in range(4):
        p1, d1 = lines[i - 1]
        p2, d2 = lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            out.append(quad[i]); continue
        ts = np.linalg.solve(A, p2 - p1)
        c = p1 + ts[0] * d1
        out.append(c if np.linalg.norm(c - quad[i]) < 2.5 * r else quad[i])
    return order_points(np.array(out, np.float32))

def enhance(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "gray":
        return cv2.cvtColor(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
    if mode == "bw":
        bs = max(31, (min(img.shape[:2]) // 25) | 1)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, 15)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    if mode == "color":
        small = cv2.dilate(cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), np.ones((5, 5), np.uint8))
        bg = cv2.medianBlur(small, max(15, (min(small.shape[:2]) // 3) | 1))
        bg = cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        return cv2.divide(img, bg, scale=255)
    raise ValueError(mode)

def select_corners(img: np.ndarray, title: str = "Clique nos 4 cantos (ENTER confirma, R reinicia, ESC cancela)"):
    scale = min(1.0, 1100.0 / max(img.shape[:2]))
    view0 = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    pts: list[tuple[int, int]] = []

    def redraw():
        v = view0.copy()
        for i, p in enumerate(pts):
            cv2.circle(v, p, 6, (0, 255, 0), -1)
            cv2.putText(v, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if len(pts) == 4:
            cv2.polylines(v, [np.array(pts)], True, (0, 255, 0), 2)
        cv2.imshow(title, v)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y)); redraw()

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    redraw()
    result = None
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        if k in (ord("r"), ord("R")):
            pts.clear(); redraw()
        if k in (13, 10) and len(pts) == 4:
            result = order_points(np.array(pts, dtype=np.float32) / scale)
            break
    cv2.destroyAllWindows()
    return result


def draw_quad(img: np.ndarray, quad: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    vis = img.copy()
    t = max(2, int(max(img.shape[:2]) / 400))
    cv2.polylines(vis, [quad.astype(np.int32)], True, color, t)
    for i, p in enumerate(quad):
        cv2.circle(vis, tuple(int(v) for v in p), t * 3, (0, 0, 255), -1)
        cv2.putText(vis, ["TL", "TR", "BR", "BL"][i], (int(p[0]) + 10, int(p[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, t * 0.45, (255, 0, 0), t)
    return vis


def save_image(path: Path, img: np.ndarray, dpi: float = 150.0) -> None:
    if path.suffix.lower() == ".pdf":
        from PIL import Image
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(str(path), "PDF", resolution=dpi)
    else:
        cv2.imwrite(str(path), img)

def scan(img, quad=None, aspect=None, paper=None, focal=None, mode="color", out_width=None,
         interp="cubic", rotate=0, refine=True, trim=0.005):
    if quad is None:
        quad = detect_document(img, refine=refine)
        if quad is None:
            return None, None, None
    if aspect is None and paper is not None:
        aspect = snap_orientation(paper, auto_aspect(quad, img.shape, focal))
    out, H = rectify(img, quad, aspect, out_width, interp, focal)
    out = enhance(trim_border(out, trim), mode)
    if rotate:
        out = cv2.rotate(out, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotate])
    return out, H, quad


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="foto do documento")
    ap.add_argument("-o", "--output", help="saída (.png/.jpg/.pdf); padrão: <entrada>_scan.png")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--corners", help='cantos "x1,y1;x2,y2;x3,y3;x4,y4" (qualquer ordem)')
    g.add_argument("--select", action="store_true", help="escolher os cantos com o mouse")
    ap.add_argument("--paper", choices=list(PAPERS), help="formato conhecido (retrato/paisagem escolhidos automaticamente)")
    ap.add_argument("--aspect", default="auto", help="largura/altura real: 'auto', 1.414, 210:297 ...")
    ap.add_argument("--focal", type=float, help="distância focal em px, se conhecida (ajuda no aspecto automático)")
    ap.add_argument("--enhance", choices=["none", "gray", "bw", "color"], default="color")
    ap.add_argument("--interp", choices=list(INTERP), default="cubic")
    ap.add_argument("--width", type=int, help="largura da saída em px (padrão: tamanho medido na foto)")
    ap.add_argument("--trim", type=float, default=0.5, help="apara esta %% da largura em cada borda (padrão 0,5)")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="gira a saída (graus, horário)")
    ap.add_argument("--no-refine", action="store_true", help="desliga o refino dos cantos (automáticos e cliques de --select; --corners nunca é refinado)")
    ap.add_argument("--debug", action="store_true", help="salva <saida>_quad.png com os cantos usados")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    img = cv2.imread(a.input)
    if img is None:
        print(f"erro: não foi possível ler '{a.input}'", file=sys.stderr)
        return 1
    out_path = Path(a.output) if a.output else Path(a.input).with_name(f"{Path(a.input).stem}_scan.png")

    quad = None
    if a.corners:
        quad = parse_corners(a.corners)
    elif a.select:
        quad = select_corners(img)
        if quad is None:
            print("erro: cantos não definidos.", file=sys.stderr)
            return 2
        if not a.no_refine:
            quad = refine_corners(img, quad)
    else:
        quad = detect_document(img, refine=not a.no_refine)
        if quad is None:
            print("erro: documento não detectado. Use --select ou --corners.", file=sys.stderr)
            return 2
        print("[info] documento detectado automaticamente.")

    out, H, quad = scan(img, quad, parse_aspect(a.aspect), PAPERS.get(a.paper) if a.paper else None,
                        a.focal, a.enhance, a.width, a.interp, a.rotate, trim=a.trim / 100)
    dpi = max(out.shape[:2]) / 11.69 if a.paper == "a4" else 150.0
    save_image(out_path, out, dpi)
    print(f"[ok] {out_path}  ({out.shape[1]}x{out.shape[0]})")
    print("H =\n", np.array2string(H / H[2, 2], precision=5, suppress_small=True))
    if a.debug:
        dbg = out_path.with_name(out_path.stem + "_quad.png")
        cv2.imwrite(str(dbg), draw_quad(img, quad))
        print(f"[ok] {dbg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
