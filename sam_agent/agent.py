import base64
import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent

from typing import Dict, List, Union

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from vertexai import model_garden

from google.adk.runners import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.artifacts import GcsArtifactService
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService

from google.adk.tools import ToolContext, FunctionTool
from google.genai import types

import google.auth
import google.auth.transport.requests

import logging
import os
import gc
import uuid
from typing import Any
import requests
import json
import numpy as np
from typing import Dict, Any, List
from PIL import Image
import cv2
import io
import random
import math

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.genai import types as gt

# --- NEW RESOLUTION CHECK AND RESIZE ---
MAX_RESOLUTION_SIDE = 1024
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
TOTAL_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_FILES_ALLOWED = 5

# ---------------- logging setup ----------------
LOG_LEVEL = os.getenv("FILE_SIZE_AGENT_LOGLEVEL", "INFO").upper()
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
logger = logging.getLogger("file_size_agent")


try:
    artifact_service = GcsArtifactService(bucket_name="gs://demo-aitinker")
    print(f"Python GcsArtifactService initialized for bucket: {artifact_service}")

except Exception as e:
    print(f"Error initializing Python GcsArtifactService: {e}")
    artifact_service = InMemoryArtifactService()


SAM_REGION = "us-east1"  # e.g., us-central1
# --- Configuration (Replace with your actual values) ---
SAM_PROJECT_ID = "272154698075"
SAM_ENDPOINT_ID = "881300351104319488"
SAM_ENDPOINT_URL = (
    f"https://us-east1-aiplatform.googleapis.com/v1/projects/{SAM_PROJECT_ID}/"
    f"locations/{SAM_REGION}/endpoints/{SAM_ENDPOINT_ID}:predict"
)

async def segment_image_with_artifact(
    tool_context: ToolContext,
) -> str:
    """
    Calls the deployed Segment Everything (Segment Anything Model - SAM) endpoint
    to generate segmentation masks for an image.

    Returns:
        A dictionary containing the prediction results, typically including
        'masks', 'scores', and 'low_res_logits'.
    """
    print("here! segment_image_with_artifact is getting called!")

    rid = uuid.uuid4().hex[:8]
    suggested_name = uuid.uuid4()
    content = tool_context.user_content
    total_size = 0
    try:
        parts = _obtain_part_and_name(content)
        preliminary_error = _preliminary_part_checks(parts, rid)
        if preliminary_error is not None:
            return preliminary_error

        creds, project = google.auth.default()
        if creds.valid is False and creds.token is None:
            auth_req = google.auth.transport.requests.Request()
            creds.refresh(auth_req)

        for image_part, inferred_name in parts:
            mime_type = await _mime_type_from_any(image_part)
            print(f"here! mime_type: {mime_type}")
            if not mime_type.startswith("image"):
                continue

            try:
                image_bytes = await _bytes_from_any(image_part, tool_context)
                img = Image.open(io.BytesIO(image_bytes))
                width, height = img.size

                if max(width, height) > MAX_RESOLUTION_SIDE:

                    # Calculate the ratio for resizing
                    ratio = MAX_RESOLUTION_SIDE / max(width, height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)

                    # Resize the image
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    print(f"Resized image from ({width}x{height}) to ({new_width}x{new_height})")

                    fill_color=(0, 0, 0)
                    img = Image.new('RGB', (MAX_RESOLUTION_SIDE, MAX_RESOLUTION_SIDE), fill_color)

                    # Calculate the position to center the resized image
                    # (x_offset, y_offset)
                    x_offset = (MAX_RESOLUTION_SIDE - new_width) // 2
                    y_offset = (MAX_RESOLUTION_SIDE - new_height) // 2

                    img.paste(resized_img, (x_offset, y_offset))
                    print(f"Image final resolution: {img.size}")

                    resized_image_buffer = io.BytesIO()
                    # Save the image in a format supported by SAM (e.g., JPEG or PNG)
                    # Use 'format=img.format' if the original format is known to be good
                    img.save(resized_image_buffer, format='PNG')

                    # Replace the original bytes with the resized bytes
                    image_bytes = resized_image_buffer.getvalue()
            except Exception as e:
                logger.exception("[%s] Could not read current message file", rid)
                return {
                    "status": "error",
                    "message": f"Could not read current message file: {e!s}",
                    "rid": rid,
                }

            size_bytes = len(image_bytes)
            if size_bytes > MAX_BYTES:
                return {
                    "status": "error",
                    "message": f"File too large: {size_bytes} bytes (limit {MAX_BYTES}).",
                    "rid": rid,
                }

            total_size += size_bytes
            if total_size > TOTAL_MAX_BYTES:
                return {
                    "status": "error",
                    "message": f"Total file size too large: {total_size} bytes (limit {TOTAL_MAX_BYTES}).",
                    "rid": rid,
                }

            artifact_name: str = inferred_name or f"{suggested_name}.png"
            tool_context.state["artifact_name"] = artifact_name
            print(f"here! artifact_name: {artifact_name}")
            pre_process_artifact_name = f"pre_processed_{artifact_name}"
            tool_context.state["pre_process_artifact_name"] = pre_process_artifact_name
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            print("here! img.save")
            img_byte_arr = img_byte_arr.getvalue()
            part_to_save = gt.Part.from_bytes(data=img_byte_arr, mime_type='image/png')
            await tool_context.save_artifact(pre_process_artifact_name, part_to_save)
            print(f"here! pre_process_artifact_name saved")

            encoded_image = base64.b64encode(image_bytes).decode('utf-8')

            headers = {
                "Authorization": f"Bearer {creds.token}",
                "Content-Type": "application/json"
            }

            request_body = {
                "instances": [
                    {
                        "image": encoded_image,
                        "prompt_type": "point",
                        "input_points": [[50, 100]],
                        "input_labels": [1]
                    }
                ]
            }

            print(f"here!! before post")
            # --- Make the API Call ---
            response = requests.post(
                SAM_ENDPOINT_URL,
                headers=headers,
                data=json.dumps(request_body)
            )
            print(f"here!! after post")
            response.raise_for_status()

            # --- Process and Summarize Output ---
            prediction_result = response.json()

            # Assuming the model returns the mask as a base64 string under 'predictions'
            # and we save it back to the context.
            # Replace 'segmented_mask_b64' with the actual output key from your model
            predictions = prediction_result.get("predictions", [{}])[0]
            #tool_context.state["sam_masks"] = predictions
            mask_arrays = predictions.get("masks")
            tool_context.state["predictions"] = predictions
            print("here! predictions")

            output_image = draw_masks_on_image(img, mask_arrays)
            print("here! draw_masks_on_image")
            if output_image is None:
                output_image = img

            #mask_bytes = base64.b64decode(mask_data)
            output_name = f"segmented_mask_for_{artifact_name}"

            img_byte_arr = io.BytesIO()
            output_image.save(img_byte_arr, format='PNG')
            print("here! output_image.save")
            img_byte_arr = img_byte_arr.getvalue()
            part_to_save = gt.Part.from_bytes(data=img_byte_arr, mime_type=mime_type)
            await tool_context.save_artifact(output_name, part_to_save)
            print("here! save_artifact")

            return (
                f"Successfully segmented the image artifact '{artifact_name}''. "
                f"The resulting segmentation mask is saved as an artifact named: '{output_name}'."
            )

        return (
            "Cannot find the image!"
        )
    except requests.exceptions.HTTPError as e:
        print(f"Error happened: HTTPError: {e}")
        return (
            f"ERROR: HTTP error calling the endpoint. Status code: {e.response.status_code}. "
            f"Details: {e.response.text}"
        )
    except Exception as e:
        print(f"Error happened: Exception : {e}")
        return f"ERROR: An unexpected error occurred: {e}"

segment_image_with_sam_tool = FunctionTool(func= segment_image_with_artifact)


def draw_masks_on_image(original_img, mask_arrays, mask_colors=None, alpha=0.5):
    """
    Draws a list of segmentation masks on top of an original image.

    Args:
        original_img (PIL.Image).
        mask_arrays (list of np.ndarray): A list of 2D NumPy arrays (int/bool, 0s and 1s),
                                          one for each segmented object/class.
        mask_colors (list of tuple, optional): A list of RGB tuples (e.g., (255, 0, 0) for red)
                                            for each mask. If None, default colors will be used.
        alpha (float, optional): Transparency level for the mask overlay (0.0 to 1.0).
                                Defaults to 0.5.

    Returns:
        PIL.Image: The original image with masks drawn on top.
    """
    if not mask_arrays:
        return original_img # Return original if no masks

    width, height = original_img.size

    # Prepare default colors if not provided
    if mask_colors is None:
        # A set of distinct default colors
        default_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (128, 0, 0),    # Dark Red
            (0, 128, 0),    # Dark Green
            (0, 0, 128),    # Dark Blue
            (128, 128, 0),  # Olive
            (0, 128, 128),  # Teal
            (128, 0, 128)   # Purple
        ]
        # Cycle through default colors if there are more masks than colors
        mask_colors = [default_colors[i % len(default_colors)] for i in range(len(mask_arrays))]

    # Ensure mask_colors matches the number of masks
    if len(mask_colors) != len(mask_arrays):
        raise ValueError("Number of mask_colors must match number of mask_images.")

    # Create a base transparent overlay image
    # This will accumulate all colored masks
    combined_overlay = Image.new('RGBA', original_img.size, (0, 0, 0, 0))
    mask_expected_shape = (height, width)
    final_image = original_img.copy()

    for i, mask_arr in enumerate(mask_arrays):
        if not isinstance(mask_arr, np.ndarray):
            mask_arr = np.array(mask_arr)

        if mask_arr.shape != mask_expected_shape:
            # Resize the mask array to match the image size (optional, depending on model output)
            mask_img = Image.fromarray(mask_arr.astype(np.uint8) * 255, mode='L')
            mask_arr = np.array(mask_img.resize((width, height)))
            # Re-convert to 0s and 1s (or boolean)
            mask_arr = mask_arr > 0
        else:
            # Ensure it's boolean or int where 1 means positive
            mask_arr = mask_arr > 0

        # 2. Get the specific color for this mask
        r, g, b = mask_colors[i]

        # Create an RGBA array for the current mask's overlay
        # Initialize with full transparency (0, 0, 0, 0)
        overlay_array = np.zeros((height, width, 4), dtype=np.uint8)

        # Where the mask_arr is True (i.e., part of the object),
        # set the RGB channels to the mask_color and the A channel to the desired alpha.
        # Scale alpha (0.0-1.0) to 0-255
        overlay_alpha = int(255 * alpha)
        overlay_array[mask_arr] = [r, g, b, overlay_alpha]

        # 3. Convert this RGBA array back to a PIL Image
        current_mask_overlay = Image.fromarray(overlay_array, mode='RGBA')

        # 4. Alpha composite the colored mask onto the current result
        final_image = Image.alpha_composite(final_image, current_mask_overlay)

    return final_image.convert("RGB")


# ---------------- part helpers ----------------
def _obtain_part_and_name(
    content: gt.Content | None
) -> list[tuple[gt.Part | None, str | None]]:
    if not content or not content.parts:
        return []
    result: list[tuple[gt.Part | None, str | None]] = []
    for p in content.parts:
        if getattr(p, "inline_data", None) and p.inline_data and p.inline_data.data:
            result.append((p, p.inline_data.display_name))
        elif (
            getattr(p, "file_data", None) and p.file_data and getattr(p.file_data, "file_uri", None)
        ):
            result.append((p, getattr(p.file_data, "display_name", None)))
    return result


async def _mime_type_from_any(part: gt.Part) -> str:
    if getattr(part, "inline_data", None) and part.inline_data and part.inline_data.mime_type:
        return part.inline_data.mime_type
    raise ValueError("Part has no mime_type.")

async def _bytes_from_any(part: gt.Part, tool_context: ToolContext) -> bytes:
    if getattr(part, "inline_data", None) and part.inline_data and part.inline_data.data:
        return part.inline_data.data
    if (
        getattr(part, "file_data", None)
        and part.file_data
        and getattr(part.file_data, "file_uri", None)
    ):
        loader = getattr(tool_context, "load_artifact_bytes", None)
        if callable(loader):
            return await loader(part.file_data.file_uri)  # type: ignore[reportUnknownVariableType]
        raise ValueError("file_data present but tool_context.load_artifact_bytes is unavailable.")
    raise ValueError("Part has no inline_data or file_data.")


def _preliminary_part_checks(parts: list[Any], rid: str) -> dict[str, Any] | None:
    if len(parts) == 0:
        return {
            "status": "error",
            "message": "No file found in the message. Attach a image and send it in the same message.",
            "rid": rid,
        }
    if len(parts) > MAX_FILES_ALLOWED:
        return {
            "status": "error",
            "message": f"Too many files. Send at most {MAX_FILES_ALLOWED} files.",
            "rid": rid,
        }
    return None




# ---------------

def find_bbox_from_mask(mask):
    """
    Finds the bounding box (bbox) for a 2D NumPy array mask.

    The bbox is returned as a tuple: (min_row, min_col, max_row, max_col).
    These represent the inclusive boundaries of the non-zero region.
    """
    # Find the indices of all non-zero elements
    rows, cols = np.where(mask)

    # Check if there are any non-zero elements
    if rows.size == 0:
        # Return a 'null' or empty bbox if the mask is entirely zero
        return None

    # Determine the min and max row and column indices
    min_row = np.min(rows)
    max_row = np.max(rows)
    min_col = np.min(cols)
    max_col = np.max(cols)

    # The bounding box is (min_row, min_col, max_row, max_col)
    return [min_row, min_col, max_row, max_col]


async def draw_masks(
    tool_context: ToolContext,
) -> str:
    """
    Draws output masks of segment_image_with_artifact tool on top of the provided image.

    Returns:
        Two processed images for the provided masks.
    """
    try:
        artifact_name = tool_context.state["artifact_name"]
        pre_process_artifact_name = tool_context.state["pre_process_artifact_name"]
        image_part = await tool_context.load_artifact(filename=pre_process_artifact_name)
        mime_type =  await _mime_type_from_any(image_part)
        image_bytes = await _bytes_from_any(image_part, tool_context)
        image_stream = io.BytesIO(image_bytes)
        img = Image.open(image_stream)
        predictions = tool_context.state["predictions"]

        segmentations = []
        for mask in predictions.get("masks", []):
            bb = find_bbox_from_mask(mask)
            segmentations.append({
                'segmentation': mask,
                'bbox': bb
            })

        display_image = draw_sam_masks(
            img,
            segmentations,
            draw_contours=True,
            annotate=True
        )
        post_process_artifact_name = f"post_processed_{artifact_name}"
        tool_context.state["post_process_artifact_name"] = post_process_artifact_name
        display_image = Image.fromarray(display_image)
        img_byte_arr = io.BytesIO()
        display_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        part_to_save = gt.Part.from_bytes(data=img_byte_arr, mime_type='image/png')
        await tool_context.save_artifact(post_process_artifact_name, part_to_save)
        print("here! display_image.save")

        display_image_fill_masks = draw_sam_masks(
            img,
            segmentations,
            draw_contours=True,
            fill_masks=True,

            annotate=True
        )
        post_process_fill_masks_artifact_name = f"post_processed_fill_masks_{artifact_name}"
        tool_context.state["post_process_fill_masks_artifact_name"] = post_process_fill_masks_artifact_name
        display_image_fill_masks = Image.fromarray(display_image_fill_masks)
        img_byte_arr = io.BytesIO()
        display_image_fill_masks.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        part_to_save = gt.Part.from_bytes(data=img_byte_arr, mime_type='image/png')
        await tool_context.save_artifact(post_process_fill_masks_artifact_name, part_to_save)
        print("here! display_image_fill_masks.save")

        deduped_bboxes_gemini = [
            {
                'bbox_2d': [
                    int(mask['bbox'][1] / 960 * 1000),
                    int(mask['bbox'][0] / 1280 * 1000),
                    int((mask['bbox'][3] + mask['bbox'][1]) / 960 * 1000),
                    int((mask['bbox'][2] + mask['bbox'][0]) / 1280 * 1000),
                ]
            }
            for mask in segmentations
        ]

        return (
            f"Successfully draws the masks on the image '{artifact_name}''. "
            f"Display the mask is saved as an artifact named: '{post_process_artifact_name}'."
            f"Display the filled mask is saved as an artifact named: '{post_process_fill_masks_artifact_name}'."
            f"Here are the bounding boxes: '{json.dumps(deduped_bboxes_gemini)}'."
        )
    except Exception as e:
        print(f"Error happened: Exception : {e}")
        raise e
        return f"ERROR: An unexpected error occurred: {e}"


draw_masks_tool = FunctionTool(func=draw_masks )


def _to_cv_canvas(image):
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if arr.dtype != np.uint8:
        arr = cv2.convertScaleAbs(arr)
    return np.ascontiguousarray(arr)

def _rects_overlap(a, b, pad=0):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + pad <= bx0 or bx1 + pad <= ax0 or ay1 + pad <= by0 or by1 + pad <= ay0)

def _bbox_intersects(b1, b2):
    x1, y1, w1, h1 = map(int, b1)
    x2, y2, w2, h2 = map(int, b2)
    return not (x1+w1 <= x2 or x2+w2 <= x1 or y1+h1 <= y2 or y2+h2 <= y1)

def putText_outlined(img, text, org, font, scale,
                     color=(255,255,255), thickness=2,
                     outline_color=(0,0,0), outline_extra=2):
    cv2.putText(img, text, org, font, scale, outline_color, thickness + outline_extra, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)

def _top_right_corner(corners):
    pts = corners.astype(np.float32)
    y_min = np.min(pts[:, 1]); tol = 1.0
    cand = pts[pts[:, 1] <= y_min + tol]
    return cand[np.argmax(cand[:, 0])]

def _normalize_roi(roi, W, H):
    if roi is None:
        return (0, 0, W, H)
    if len(roi) != 4:
        raise ValueError("roi must be a 4-tuple")
    x0, y0, a, b = map(int, roi)
    if a <= 0 or b <= 0:  # treat as x1,y1
        rx0, ry0, rx1, ry1 = x0, y0, max(x0+1, a), max(y0+1, b)
    else:
        rx0, ry0, rx1, ry1 = x0, y0, x0 + a, y0 + b
    rx0 = max(0, min(rx0, W)); ry0 = max(0, min(ry0, H))
    rx1 = max(0, min(rx1, W)); ry1 = max(0, min(ry1, H))
    if rx1 <= rx0 or ry1 <= ry0:
        raise ValueError("roi is empty after clamping")
    return (rx0, ry0, rx1, ry1)

def draw_sam_masks(
    image, masks, *,
    draw_contours=True, draw_bboxes=False, annotate=True, fill_masks=False,
    box_alpha=0.35,          # label box transparency when fill_masks=False
    box_pad=4,               # inner padding inside label box
    base_font_scale=0.6, font_th=2,
    inner_margin=3,          # px inward from mask corner along corner->center
    outside_gap=6,           # px outward (used if needed; still requires mask overlap)
    avoidance_radii=(0, 8, 16, 24, 36),
    nudge_dirs=((1,-1),(1,0),(0,-1),(2,-1),(1,-2),(2,0),(0,-2),(1,1),(-1,-1),(-1,0),(0,1)),
    mask_edge_thickness=2,   # black boundary around filled masks
    label_edge_thickness=2,  # black boundary around label box
    max_shrink_steps=6,
    seed_colors=False,
    roi=None,                       # (x0,y0,w,h) or (x0,y0,x1,y1) for label confinement
    annotate_masks_in_roi_only=True,
    roi_label_min_coverage=0.02,    # min fraction of mask area inside ROI to draw a label
    # NEW: stable IDs
    id_key=None,                    # e.g., "id" or "mask_id"; if present in mask dict it will be used
    global_mask_list=None           # pass the full original masks list to preserve original indices
):
    """
    ROI-aware labeling with coverage threshold + STABLE IDS:
      - If roi is provided, labels are confined to it.
      - Only label masks whose bbox intersects ROI (if annotate_masks_in_roi_only=True) AND whose
        (mask_area_inside_roi / mask_area_total) >= roi_label_min_coverage.
      - Label text uses the original mask ID:
          1) mask[id_key] if provided and present,
          2) else index in global_mask_list (by object identity),
          3) else local enumerate index (fallback).
    """
    display_image = _to_cv_canvas(image)
    H, W = display_image.shape[:2]
    rx0, ry0, rx1, ry1 = _normalize_roi(roi, W, H) if roi is not None else (0, 0, W, H)
    rng = random.Random(0) if seed_colors else random
    print("here! rng")
    font = cv2.FONT_HERSHEY_SIMPLEX
    print("here! font")

    # Build a fast map from mask object -> original index if provided
    global_index_map = None
    if global_mask_list is not None:
        global_index_map = {id(m): i for i, m in enumerate(global_mask_list)}
    print("here! global_mask_list ")

    def _resolve_label_id(mask_obj, local_idx):
        if id_key is not None and isinstance(mask_obj, dict) and id_key in mask_obj:
            return mask_obj[id_key]
        if global_index_map is not None:
            gi = global_index_map.get(id(mask_obj))
            if gi is not None:
                return gi
        return local_idx  # fallback (subset-only behavior)

    # -------- Precompute per-mask data (and stable label IDs) --------
    items = []
    for local_i, m in enumerate(masks):
        # Skip masks outside ROI for labeling if requested
        if roi is not None and annotate_masks_in_roi_only:
            if not _bbox_intersects(m['bbox'], (rx0, ry0, rx1 - rx0, ry1 - ry0)):
                continue

        if not isinstance(m, dict):
            continue
        seg = np.asarray(m['segmentation'], dtype=bool)
        if seg.shape[:2] != (H, W):
            seg = cv2.resize(seg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        seg_bin = np.ascontiguousarray(seg.astype(np.uint8))
        seg_255 = np.ascontiguousarray(seg_bin * 255)

        contours, _ = cv2.findContours(seg_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        areas_cont = [cv2.contourArea(c) for c in contours]
        largest_contour = contours[int(np.argmax(areas_cont))]
        rrect = cv2.minAreaRect(largest_contour)
        corners = cv2.boxPoints(rrect).astype(np.float32)
        rect_center = np.array(rrect[0], dtype=np.float32)
        tr_corner = _top_right_corner(corners)

        mask_area = int(m.get('area', int(np.count_nonzero(seg_bin))))
        # ROI coverage for labeling
        if roi is not None and mask_area > 0:
            roi_crop = seg_bin[ry0:ry1, rx0:rx1]
            inside_pixels = int(np.count_nonzero(roi_crop))
            coverage = inside_pixels / float(mask_area)
        else:
            coverage = 1.0

        color = (rng.randint(100,255), rng.randint(100,255), rng.randint(100,255))
        x, y, w, h = map(int, m['bbox'])
        label_id = _resolve_label_id(m, local_i)

        items.append({
            "label_id": label_id,            # <<< stable id for text
            "contours": contours,
            "largest_contour": largest_contour,
            "seg_bin": seg_bin,
            "corners": corners,
            "tr_corner": tr_corner,
            "rect_center": rect_center,
            "bbox": (x, y, w, h),
            "color": color,
            "area": mask_area,
            "label_eligible": (coverage >= roi_label_min_coverage)
        })

    # -------- Draw masks (within the filtered set) --------
    print("here! Draw masks")
    if fill_masks:
        order = sorted(items, key=lambda it: it["area"], reverse=True)
        for it in order:
            cv2.drawContours(display_image, it["contours"], -1, it["color"], cv2.FILLED)
        for it in order:
            cv2.drawContours(display_image, it["contours"], -1, (0,0,0), mask_edge_thickness)
    elif draw_contours:
        for it in items:
            cv2.drawContours(display_image, it["contours"], -1, it["color"], 2)

    if draw_bboxes:
        for it in items:
            x, y, w, h = it["bbox"]
            cv2.rectangle(display_image, (x, y), (x+w, y+h), it["color"], 2)

    if not annotate:
        return display_image

    # -------- Helpers honoring ROI --------
    def _rect_from_center(cx, cy, box_w, box_h):
        minx, miny, maxx, maxy = rx0, ry0, rx1, ry1
        x0 = int(round(cx - box_w/2)); y0 = int(round(cy - box_h/2))
        x0 = max(minx, min(x0, maxx - box_w))
        y0 = max(miny, min(y0, maxy - box_h))
        return (x0, y0, x0 + box_w, y0 + box_h)

    def _inside_center_from_corner(corner_xy, center_xy, box_w, box_h):
        ax, ay = float(corner_xy[0]), float(corner_xy[1])
        vx = center_xy[0] - ax; vy = center_xy[1] - ay
        n = math.hypot(vx, vy) or 1.0
        ux, uy = vx/n, vy/n
        half_diag = 0.5 * math.hypot(box_w, box_h)
        cx = ax + ux * (half_diag + inner_margin)
        cy = ay + uy * (half_diag + inner_margin)
        return cx, cy

    def _rect_intersects_mask(rect, seg_bin):
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0: return False
        roi_m = seg_bin[y0:y1, x0:x1]
        return roi_m.size > 0 and np.any(roi_m)

    # -------- Candidate generation (ROI-constrained; coverage-gated) --------
    label_specs = []
    for it in items:
        lb_id = it["label_id"]
        seg_bin = it["seg_bin"]
        tr = it["tr_corner"]
        center_xy = it["rect_center"]
        color = it["color"]

        candidates = []
        if it["label_eligible"]:
            text = str(lb_id)
            scale = base_font_scale
            for s_step in range(max_shrink_steps):
                font_scale = scale * (0.9 ** s_step)
                (tw, th_pix), baseline = cv2.getTextSize(text, font, font_scale, font_th)
                box_w = tw + 2*box_pad
                box_h = th_pix + baseline + 2*box_pad

                cx_in, cy_in = _inside_center_from_corner(tr, center_xy, box_w, box_h)

                centers = [(cx_in, cy_in)]
                for r in avoidance_radii:
                    if r == 0:
                        continue
                    for dx, dy in nudge_dirs:
                        centers.append((cx_in + dx*r, cy_in + dy*r))
                # deeper toward center
                vx = center_xy[0] - tr[0]; vy = center_xy[1] - tr[1]
                n = math.hypot(vx, vy) or 1.0
                ux, uy = vx/n, vy/n
                for extra in (8, 16, 24):
                    centers.append((cx_in + ux*extra, cy_in + uy*extra))

                for (cx, cy) in centers:
                    rect = _rect_from_center(cx, cy, box_w, box_h)
                    if not _rect_intersects_mask(rect, seg_bin):
                        continue
                    x0, y0, x1, y1 = rect
                    Lh = y1 - y0
                    y_offset = (Lh - (th_pix + baseline)) // 2
                    org = (x0 + box_pad, y0 + y_offset + th_pix)
                    dist2 = (cx - cx_in)**2 + (cy - cy_in)**2
                    candidates.append((rect, org, font_scale, dist2))

                if len(candidates) >= 30:
                    break

            candidates.sort(key=lambda t: t[3])

        label_specs.append({"label_id": lb_id, "color": color, "seg_bin": seg_bin, "candidates": candidates})

    # Big masks pick first
    order_for_labels = sorted(range(len(items)), key=lambda k: items[k]["area"], reverse=True)

    # -------- Greedy packing + conflict resolution --------
    assignments = {k: None for k in order_for_labels}
    chosen_rects = {}

    for k in order_for_labels:
        cand_list = label_specs[k]["candidates"]
        for ci, (rect, _, _, _) in enumerate(cand_list):
            if all(not _rects_overlap(rect, chosen_rects[j]) for j in chosen_rects):
                assignments[k] = ci
                chosen_rects[k] = rect
                break

    def remaining_options(k, exclude_rects):
        curr = assignments[k]
        cands = label_specs[k]["candidates"]
        cnt = 0
        opt = None
        for ci, (rect, *_rest) in enumerate(cands):
            if ci == curr:
                continue
            if any(_rects_overlap(rect, r) for r in exclude_rects):
                continue
            cnt += 1
            if opt is None: opt = ci
        return cnt, opt

    for _ in range(100):
        ks = list(chosen_rects.keys())
        pairs = []
        for i in range(len(ks)):
            for j in range(i+1, len(ks)):
                a, b = ks[i], ks[j]
                if _rects_overlap(chosen_rects[a], chosen_rects[b]):
                    pairs.append((a, b))
        unassigned = [k for k in order_for_labels if assignments[k] is None]

        if not pairs and not unassigned:
            break

        progress = False

        for k in list(unassigned):
            cands = label_specs[k]["candidates"]
            for ci, (rect, *_r) in enumerate(cands):
                blockers = [j for j, r in chosen_rects.items() if _rects_overlap(rect, r)]
                if not blockers:
                    assignments[k] = ci; chosen_rects[k] = rect; progress = True; break
                best_j, best_ci, best_count = None, None, -1
                for j in blockers:
                    others = [r for q, r in chosen_rects.items() if q != j]
                    cnt, alt_ci = remaining_options(j, others + [rect])
                    if cnt > best_count and alt_ci is not None:
                        best_j, best_ci, best_count = j, alt_ci, cnt
                if best_j is not None and best_ci is not None:
                    assignments[best_j] = best_ci
                    chosen_rects[best_j] = label_specs[best_j]["candidates"][best_ci][0]
                    assignments[k] = ci
                    chosen_rects[k] = rect
                    progress = True
                    break

        if progress:
            continue

        moved = False
        for a, b in pairs:
            others_for_a = [r for q, r in chosen_rects.items() if q not in (a,)]
            others_for_b = [r for q, r in chosen_rects.items() if q not in (b,)]
            cnt_a, alt_a = remaining_options(a, others_for_a)
            cnt_b, alt_b = remaining_options(b, others_for_b)
            target, alt_ci = (a, alt_a) if cnt_a > cnt_b else (b, alt_b)
            if alt_ci is not None:
                assignments[target] = alt_ci
                chosen_rects[target] = label_specs[target]["candidates"][alt_ci][0]
                moved = True
        if not moved:
            break

    # -------- Render labels (ROI-confined) --------
    effective_alpha = 1.0 if fill_masks else float(np.clip(box_alpha, 0.0, 1.0))

    for k in order_for_labels:
        cands = label_specs[k]["candidates"]
        if not cands:
            continue
        ci = assignments.get(k, None)
        if ci is None:
            continue
        rect, org, font_scale, _d = cands[ci]
        color = label_specs[k]["color"]
        text  = str(label_specs[k]["label_id"])

        x0, y0, x1, y1 = rect
        roi_box = display_image[y0:y1, x0:x1]
        bg = np.empty_like(roi_box); bg[...] = color
        roi_box[:] = (roi_box.astype(np.float32) * (1.0 - effective_alpha) + bg.astype(np.float32) * effective_alpha).astype(np.uint8)

        if x1 > x0 and y1 > y0:
            cv2.rectangle(display_image, (x0, y0), (x1-1, y1-1), (0,0,0), label_edge_thickness)

        putText_outlined(display_image, text, org, font, font_scale,
                         color=(255,255,255), thickness=font_th,
                         outline_color=(0,0,0), outline_extra=2)

    return display_image
# ---------------------

agent = Agent(
    name="segment_image",
    model="gemini-2.0-flash",
    description=(
        "Agent to segment images."
    ),
    instruction=(
        "You are a expert visual reasoning agent who can answer user questions about segmentation of an image in PNG, JPG, or WEBP formats."
        "You are an "
        ""
        "Your task is to answer user queries with as much accuracy and rigor as possible."
        "For each image analysis query, you must run the segment_image_with_sam_tool tool to receive the image with object boundaries and object ids outlined, and another same image with object masks filled with color. It will also return a  list of bounding boxes corresponding to the objects drawn on the image (Object id corresponding to the index in the list). The bounding boxes follow the following format: [y1, x1, x2, y2] normalized to 0-1000 range"
        "You must rely on this evidence when answering complex queries, especially those involving counting"
        "and / or reasoning about complex properties."
        "You may use your own visual perception of the image (for example, to classify the object corresponding to the mask), but you should always align with the bounding boxes given by the the 'REPLACE WITH FUNC NAME' tool."
        ""
        "Here is the workflow you should follow:"
        "1) Carefully analyze the user query and the input image (if provided)"
        "2) If the user query involves image analysis, execute the segment_image_with_sam_tool tool to receive the information about the objects visible on the image"
        "3) Perform analysis using all available means. You may execute code when needed"
        "4) Following the conducted analysis, provide a final comprehensive answer to the user query."
        "5) Display all the analyzed images to the user."
    ),
    tools=[segment_image_with_sam_tool, draw_masks_tool ],
)

root_agent=agent

# Provide it to the Runner
runner = Runner(
    agent=agent,
    app_name="segment any image",
    session_service=InMemorySessionService(),
    artifact_service=artifact_service # Service must be provided here
)
