import os
import re
from abc import ABC, abstractmethod
from itertools import chain
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
import torch
from doclayout_yolo import YOLOv10
from pdfplumber.page import Page

from utils import get_config_value

from .schema import BoundingBox, PageElement

IMAGE_PATH_TEMPLATE = "page_{page_index}_image_{image_index}.png"
IMAGE_LINK_TEMPLATE = "![page_{page_index}_image_{i}]({link})"


class BaseExtractor(ABC):
    """Base Extractor"""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def _execute(self, page: Page, page_index: int, **kwargs) -> Any:
        pass

    def execute(self, page: Page, page_index: int, **kwargs) -> Any:
        return self._execute(page, page_index, **kwargs)


class ImageExtractor(BaseExtractor):
    """Extracts images from a page."""

    def __init__(self, config: dict, image_output_dir: Path, markdown_output_dir: Path):
        super().__init__(config)
        # Auto find cuda
        self.yolo_model = YOLOv10(config["doclayout_yolo"]["model_path"]).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.image_output_dir = image_output_dir  # Image output path
        self.markdown_output_dir = markdown_output_dir  # Markdown output path

        # 从配置读取参数
        self.yolo_predict_resolution = get_config_value(
            config, "image_extractor", "yolo_predict_resolution", 180, int
        )
        self.yolo_imgsz = get_config_value(
            config, "image_extractor", "yolo_imgsz", 1024, int
        )
        self.yolo_conf = get_config_value(
            config, "image_extractor", "yolo_conf", 0.65, float
        )
        self.yolo_image_class_id = get_config_value(
            config, "image_extractor", "yolo_image_class_id", 3, int
        )
        self.yolo_verbose = get_config_value(
            config, "image_extractor", "yolo_verbose", False, bool
        )
        self.save_dpi = get_config_value(
            config, "image_extractor", "save_dpi", 720, int
        )
        self.overlap_thresh = get_config_value(
            config, "image_extractor", "overlap_thresh", 0.75, float
        )
        self.transfer_dpi = get_config_value(
            config, "image_extractor", "transfer_dpi", 180, int
        )

    def _execute(
        self,
        page: Page,
        page_index: int,
        save_dpi: int = None,
        overlap_thresh: float = None,
    ) -> List[PageElement]:
        """
        get image elements from page
        """
        # 使用配置中的默认值，允许调用时覆盖
        if save_dpi is None:
            save_dpi = self.save_dpi
        if overlap_thresh is None:
            overlap_thresh = self.overlap_thresh

        image_elements: List[PageElement] = []
        result = self.yolo_model.predict(
            source=page.to_image(resolution=self.yolo_predict_resolution).original,
            imgsz=self.yolo_imgsz,
            conf=self.yolo_conf,
            verbose=self.yolo_verbose,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )[0]
        indices = torch.where(result.boxes.cls == self.yolo_image_class_id)[0]

        if indices.numel() == 0:
            return []
        dpi_scale = (
            save_dpi / self.yolo_predict_resolution
        )  # scale for converting to save_dpi
        boxes = [tuple(b.item() for b in box) for box in result.boxes.xyxy[indices]]
        merged_boxes = self.merge_overlapping_boxes(
            boxes=boxes, overlap_thresh=overlap_thresh
        )
        # Save high resolution image
        high_res_image = page.to_image(resolution=save_dpi).original
        for i, box in enumerate(merged_boxes):
            save_path = Path(self.image_output_dir) / IMAGE_PATH_TEMPLATE.format(
                page_index=page_index, image_index=i
            )
            high_res_image.crop(tuple(b * dpi_scale for b in box)).save(save_path)
            relative_image_path = os.path.relpath(
                save_path, self.markdown_output_dir.parent
            )
            link = Path(relative_image_path).as_posix()
            image_elements.append(
                PageElement(
                    content_type="figure",
                    page_idx=page_index,
                    bbox=self.base_transfer(bbox=box, page=page, dpi=self.transfer_dpi),
                    content=IMAGE_LINK_TEMPLATE.format(
                        page_index=page_index, i=i, link=link
                    ),
                )
            )
        return image_elements

    @staticmethod
    def merge_overlapping_boxes(
        boxes: List[BoundingBox], overlap_thresh: float = 0.5
    ) -> List[BoundingBox]:
        """Merge overlapping boxes"""

        # If the overlap area is too high, the boxes are merged
        def _overlap_ratio(boxA: BoundingBox, boxB: BoundingBox) -> float:
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            inter_w = max(0, xB - xA)
            inter_h = max(0, yB - yA)
            inter_area = inter_w * inter_h
            if inter_area == 0:
                return 0.0
            areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            # Return the ratio of the overlap area to the smaller box area
            return inter_area / min(areaA, areaB)

        merged = []
        boxes = boxes.copy()
        while boxes:
            base = boxes.pop(0)
            new_boxes = []
            for b in boxes:
                if _overlap_ratio(base, b) > overlap_thresh:
                    # Merge
                    base = (
                        min(base[0], b[0]),
                        min(base[1], b[1]),
                        max(base[2], b[2]),
                        max(base[3], b[3]),
                    )
                else:
                    new_boxes.append(b)
            merged.append(base)
            boxes = new_boxes
        return merged

    @staticmethod
    def base_transfer(bbox: BoundingBox, page: Page, dpi: int = 180) -> BoundingBox:
        """Convert image bbox coordinates to PDF space coordinates"""
        x1, y1, x2, y2 = bbox
        pdf_w, pdf_h = float(page.width), float(page.height)
        scale_x = pdf_w / (pdf_w * dpi / 72.0)
        scale_y = pdf_h / (pdf_h * dpi / 72.0)
        return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


class TextExtractor(BaseExtractor):
    """Extract text content"""

    def __init__(self, config: dict = None, header_predict_work: bool = False):
        if config is None:
            config = {}
        super().__init__(config)
        self.work = header_predict_work
        # 从配置读取参数
        self.x_tolerance = get_config_value(
            config, "text_extractor", "x_tolerance", 1, int
        )
        self.y_tolerance = get_config_value(
            config, "text_extractor", "y_tolerance", 1, int
        )
        self.keep_blank_chars = get_config_value(
            config, "text_extractor", "keep_blank_chars", False, bool
        )

    def _execute(self, page: Page, page_index: int) -> List[PageElement]:
        text: List[PageElement] = []
        tables, table_boxes = self.get_tables(page=page, page_index=page_index)
        links, link_boxes = self.get_links(page=page, page_index=page_index)
        plain_texts = self.get_plain_text(
            page=page, page_index=page_index, filter_bboxes=table_boxes + link_boxes
        )
        text = list(chain(tables, links, plain_texts))
        return text

    @staticmethod
    def get_tables(
        page: Page, page_index: int
    ) -> Tuple[List[PageElement], List[BoundingBox]]:
        """
        Get table elements
        Returns:
            Tuple[List[PageElement], List[Bbox]]: List of elements and their corresponding bounding boxes
        """
        table_elements: List[PageElement] = []
        bboxes: List[BoundingBox] = []
        tables = page.find_tables()
        for table in tables:
            data = table.extract()
            if not data:
                continue
            try:
                df = pd.DataFrame(data)
                if not df.empty:
                    df.columns = df.iloc[0]
                    df = df[1:]
                    df_clean = df.replace(r"^\s*$", None, regex=True).dropna(how="all")
                    if df_clean.empty:
                        continue
                md_table = df.to_markdown(index=False)
            except Exception:
                # Keep markdown style
                md_table = "\n".join([" | ".join(map(str, row)) for row in data])
            bboxes.append(table.bbox)
            table_elements.append(
                PageElement(
                    content_type="table",
                    bbox=tuple(map(float, table.bbox)),
                    page_idx=page_index,
                    content=md_table,
                )
            )
        return table_elements, bboxes

    def get_links(
        self, page: Page, page_index: int
    ) -> Tuple[List[PageElement], List[BoundingBox]]:
        """Get link elements"""
        annots = getattr(page, "annots", None)
        if not annots:
            return [], []
        link_elements: List[PageElement] = []
        bboxes: List[BoundingBox] = []
        words = (
            page.extract_words(
                x_tolerance=self.x_tolerance,
                y_tolerance=self.y_tolerance,
                keep_blank_chars=self.keep_blank_chars,
            )
            or []
        )
        for annot in annots:
            subtype = (annot.get("Subtype") or "").lower()
            if "link" not in subtype:
                continue
            uri = annot.get("URI") or (annot.get("A") or {}).get("URI")
            rect = annot.get("Rect")
            if not rect:
                continue
            bbox = tuple(rect)
            in_words = [
                w
                for w in words
                if w["x0"] >= bbox[0]
                and w["x1"] <= bbox[2]
                and w["top"] >= bbox[1]
                and w["bottom"] <= bbox[3]
            ]
            text = " ".join(w.get("text", "") for w in in_words)
            content = f"[{text}]({uri})" if uri else text
            bboxes.append(bbox)
            link_elements.append(
                PageElement(
                    content_type="link", bbox=bbox, content=content, page_idx=page_index
                )
            )

        return link_elements, bboxes

    # TODO: Only support single column for now, will be modified to support multiple columns later
    def get_plain_text(
        self, page: Page, page_index: int, filter_bboxes: List[BoundingBox]
    ) -> List[PageElement]:
        """Extract plain text from page (only consider single column)"""
        lines = page.extract_text_lines()
        if not lines:
            return []
        text_elements: List[PageElement] = []
        # Filter areas to prevent duplicate extraction of text in tables/links
        for line in lines:
            if any(
                x0 <= line["x0"]
                and line["x1"] <= x1
                and y0 <= line["top"]
                and line["bottom"] <= y1
                for (x0, y0, x1, y1) in filter_bboxes
            ):
                continue
            is_header, content = self.header_predict(text=line["text"])
            text_elements.append(
                PageElement(
                    content_type="header" if is_header else "plain_text",
                    bbox=(line["x0"], line["top"], line["x1"], line["bottom"]),
                    content=content,
                    page_idx=page_index,
                )
            )
        return text_elements

    def header_predict(self, text: str) -> Tuple[bool, str]:
        clean = text.strip()
        if not self.work:
            return False, clean
        if not clean:
            return False, clean
        header_regex = re.compile(
            r"^(?!"  # Negative lookahead to exclude directory page number format
            r".*[.·]{2,}\s*\d+$"
            r")"
            r"(?:"
            r"(?P<allcaps>[A-Z\s\d]+$)"  # All capital letters
            r"|第(?P<zh_num>[一二三四五六七八九十百千万]+)(?P<zh_unit>[章节])"  # Chinese chapter
            r"|(?P<num>\d+(?:\.\d+)*\s*\S+)"  # Number + title
            r"|(?P<trailing_colon>.+:$)"  # Colon ending
            r")"
        )
        m = header_regex.match(clean)
        if not m:
            return False, clean
        level = 1
        if m.group("zh_num"):
            if m.group("zh_unit") == "章":
                level = 1
            else:
                level = 2
        elif m.group("num"):
            parts = clean.split()[0].split(".")
            level = len(parts)
        elif m.group("allcaps"):
            level = 1
        elif m.group("trailing_colon"):
            level = 2
        return True, f"{'#' * level} {clean}"
