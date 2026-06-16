#!/usr/bin/env python3
"""Process a newly uploaded literature PDF into:
- page-aligned Chinese translation PDF
- website JSON with full page translations and critical reading fields
- updated data/library.json

This script is designed for GitHub Actions. It requires OPENAI_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
from openai import OpenAI
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def slugify(text: str) -> str:
    text = (text or "paper").lower()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"doi\.org/", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text[:80] or "paper"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def extract_pdf(pdf_path: Path) -> Dict[str, Any]:
    doc = fitz.open(pdf_path)
    pages: List[Dict[str, Any]] = []
    for idx, page in enumerate(doc, start=1):
        width, height = float(page.rect.width), float(page.rect.height)
        raw_blocks = page.get_text("blocks")
        blocks = []
        page_text_parts = []
        for b in raw_blocks:
            x0, y0, x1, y1, text = b[:5]
            text = clean_text(text)
            if not text:
                continue
            # Drop isolated page numbers but keep normal content.
            if len(text) < 4 and re.fullmatch(r"\d+", text):
                continue
            blocks.append({
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "text": text,
            })
            page_text_parts.append(text)
        pages.append({
            "page": idx,
            "width": width,
            "height": height,
            "text": "\n".join(page_text_parts),
            "blocks": blocks,
        })
    meta = doc.metadata or {}
    doc.close()
    return {"metadata": meta, "pages": pages, "pageCount": len(pages)}


def ask_json(client: OpenAI, model: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        text={
            "format": {
                "type": "json_schema",
                "name": "literature_processing",
                "strict": True,
                "schema": schema,
            }
        },
    )
    raw = response.output_text
    return json.loads(raw)


TRANSLATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "page": {"type": "integer"},
                    "translation": {"type": "string"},
                    "keyPoints": {"type": "array", "items": {"type": "string"}},
                    "figureNote": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["page", "translation", "keyPoints", "figureNote", "evidence"],
            },
        }
    },
    "required": ["pages"],
}

ANALYSIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "titleCn": {"type": "string"},
        "journal": {"type": "string"},
        "year": {"type": "string"},
        "doi": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "criticalAppraisal": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "researchQuestion": {"type": "string"},
                "contribution": {"type": "array", "items": {"type": "string"}},
                "keyFindings": {"type": "array", "items": {"type": "string"}},
                "methods": {"type": "array", "items": {"type": "string"}},
                "statistics": {"type": "string"},
                "reliability": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}},
                "contradictions": {"type": "array", "items": {"type": "string"}},
                "possibleReasons": {"type": "string"},
                "transferability": {"type": "array", "items": {"type": "string"}},
                "nextExperiments": {"type": "array", "items": {"type": "string"}},
                "riskLevel": {"type": "string"},
                "qualityScore": {"type": "string"},
            },
            "required": [
                "researchQuestion", "contribution", "keyFindings", "methods", "statistics",
                "reliability", "limitations", "contradictions", "possibleReasons",
                "transferability", "nextExperiments", "riskLevel", "qualityScore"
            ],
        },
    },
    "required": ["title", "titleCn", "journal", "year", "doi", "tags", "summary", "criticalAppraisal"],
}


def translate_pages(client: OpenAI, model: str, pages: List[Dict[str, Any]], batch_size: int = 2) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for i in range(0, len(pages), batch_size):
        batch = pages[i:i + batch_size]
        payload = [
            {"page": p["page"], "text": p["text"][:14000]}
            for p in batch
        ]
        prompt = (
            "你是冶金/矿物加工方向的专业学术翻译。请逐页把英文论文翻译成中文。\n"
            "要求：1) 保留术语准确性；2) 不省略正文；3) 页码一一对应；"
            "4) keyPoints 提取本页关键点；5) figureNote 简述本页图表/表格；"
            "6) evidence 摘录少量英文原句作为依据。\n"
            "返回严格 JSON。待处理页面如下：\n" + json.dumps(payload, ensure_ascii=False)
        )
        result = ask_json(client, model, prompt, TRANSLATION_SCHEMA)
        for item in result.get("pages", []):
            out[int(item["page"])] = item
        print(f"translated pages {batch[0]['page']}-{batch[-1]['page']}", flush=True)
    return out


def analyze_paper(client: OpenAI, model: str, extracted: Dict[str, Any], translations: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    first_text = "\n\n".join(p["text"] for p in extracted["pages"][:4])[:50000]
    all_key_points = []
    for p in translations.values():
        all_key_points.extend(p.get("keyPoints", []))
    prompt = (
        "你是低品位锰矿、铁锰矿分离、还原焙烧、磁选、酸浸、MnSO4 溶液制备方向的论文精读助手。\n"
        "请根据论文首页/摘要/正文和逐页关键点，生成网站需要的文献信息与批判性精读。\n"
        "重点回答：研究问题、核心贡献、结论可靠性、样本量/统计/实验设计、局限、与同领域是否矛盾、矛盾原因、能否迁移到低品位锰矿实验路线。\n"
        "返回严格 JSON。\n"
        f"PDF metadata: {json.dumps(extracted.get('metadata', {}), ensure_ascii=False)}\n"
        f"Extracted first pages: {first_text}\n"
        f"Page key points: {json.dumps(all_key_points[:80], ensure_ascii=False)}\n"
    )
    return ask_json(client, model, prompt, ANALYSIS_SCHEMA)


def wrap_cjk(text: str, max_chars: int) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    lines: List[str] = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        while para:
            lines.append(para[:max_chars])
            para = para[max_chars:]
    return lines or [""]


def draw_wrapped(c: canvas.Canvas, text: str, x: float, y_top: float, width: float, page_h: float, font_size: float = 9) -> None:
    max_chars = max(8, int(width / (font_size * 0.55)))
    lines = wrap_cjk(text, max_chars)
    y = page_h - y_top
    c.setFont("STSong-Light", font_size)
    c.setFillColor(HexColor("#263238"))
    for line in lines[:28]:
        c.drawString(x, y, line)
        y -= font_size * 1.35
        if y < 30:
            break


def build_translation_pdf(extracted: Dict[str, Any], translations: Dict[int, Dict[str, Any]], out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_pdf))
    for page in extracted["pages"]:
        n = int(page["page"])
        w, h = float(page["width"]), float(page["height"])
        c.setPageSize((w, h))
        c.setFillColor(HexColor("#ffffff"))
        c.rect(0, 0, w, h, fill=1, stroke=0)
        c.setFont("STSong-Light", 10)
        c.setFillColor(HexColor("#2f6f5e"))
        c.drawString(28, h - 24, f"第 {n} 页 中文译文")
        trans = translations.get(n, {})
        text = trans.get("translation", "")
        blocks = page.get("blocks", [])
        # Simple page-corresponding layout: place translated paragraphs near original block positions.
        paras = [p for p in re.split(r"(?<=[。！？；])", text) if p.strip()]
        if blocks and paras:
            for idx, b in enumerate(blocks[:len(paras)]):
                x0, y0, x1, y1 = b["bbox"]
                draw_wrapped(c, paras[idx], max(28, x0), max(34, y0), max(120, x1 - x0), h, 8.5)
        else:
            y = 54
            for line in wrap_cjk(text, 46):
                draw_wrapped(c, line, 36, y, w - 72, h, 10)
                y += 16
        c.setFont("STSong-Light", 8)
        c.setFillColor(HexColor("#98a2b3"))
        c.drawCentredString(w / 2, 18, str(n))
        c.showPage()
    c.save()


def build_site_json(pdf_path: Path, extracted: Dict[str, Any], translations: Dict[int, Dict[str, Any]], analysis: Dict[str, Any], remote_id: str) -> Dict[str, Any]:
    page_translations = {}
    page_figures = {}
    page_key_points = {}
    original_evidence = {}
    for n, item in translations.items():
        page_translations[str(n)] = [item.get("translation", "")]
        page_figures[str(n)] = item.get("figureNote", "")
        page_key_points[str(n)] = item.get("keyPoints", [])
        original_evidence[str(n)] = item.get("evidence", [])
    return {
        "id": remote_id,
        "remoteId": remote_id,
        "title": analysis.get("title", pdf_path.stem),
        "titleCn": analysis.get("titleCn", pdf_path.stem),
        "journal": analysis.get("journal", ""),
        "year": analysis.get("year", ""),
        "doi": analysis.get("doi", ""),
        "journalRank": {"IF": "-", "CAS": "-", "JCR": "-"},
        "tags": analysis.get("tags", []),
        "pageCount": extracted["pageCount"],
        "assets": {
            "sourcePath": f"papers/{remote_id}-en.pdf",
            "translationPath": f"papers/{remote_id}-cn.pdf",
            "jsonPath": f"data/{remote_id}.json",
        },
        "pageTranslations": page_translations,
        "pageFigures": page_figures,
        "pageKeyPoints": page_key_points,
        "originalEvidence": original_evidence,
        "aiReading": {
            "summary": analysis.get("summary", ""),
            "criticalAppraisal": analysis.get("criticalAppraisal", {}),
        },
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }


def update_library(doc_json: Dict[str, Any], library_path: Path) -> None:
    library_path.parent.mkdir(parents=True, exist_ok=True)
    if library_path.exists():
        data = json.loads(library_path.read_text(encoding="utf-8"))
    else:
        data = {"version": 1, "docs": []}
    if isinstance(data, list):
        data = {"version": 1, "docs": data}
    docs = data.get("docs", [])
    slim = {k: doc_json[k] for k in ["id", "remoteId", "title", "titleCn", "journal", "year", "doi", "journalRank", "tags", "pageCount", "assets"] if k in doc_json}
    replaced = False
    for i, old in enumerate(docs):
        if old.get("id") == doc_json["id"] or old.get("remoteId") == doc_json["remoteId"]:
            docs[i] = slim
            replaced = True
            break
    if not replaced:
        docs.insert(0, slim)
    data["docs"] = docs
    data["updatedAt"] = datetime.now(timezone.utc).isoformat()
    library_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--library", type=Path, default=Path("data/library.json"))
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is missing. Add it as a repository Actions secret.", file=sys.stderr)
        return 2
    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    client = OpenAI()
    remote_id = slugify(args.pdf.stem)
    source_out = Path("papers") / f"{remote_id}-en.pdf"
    trans_out = Path("papers") / f"{remote_id}-cn.pdf"
    json_out = Path("data") / f"{remote_id}.json"
    source_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)

    extracted = extract_pdf(args.pdf)
    translations = translate_pages(client, args.model, extracted["pages"])
    analysis = analyze_paper(client, args.model, extracted, translations)
    build_translation_pdf(extracted, translations, trans_out)
    doc_json = build_site_json(args.pdf, extracted, translations, analysis, remote_id)
    json_out.write_text(json.dumps(doc_json, ensure_ascii=False, indent=2), encoding="utf-8")
    source_out.write_bytes(args.pdf.read_bytes())
    update_library(doc_json, args.library)
    print(f"Generated: {source_out}, {trans_out}, {json_out}, {args.library}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
