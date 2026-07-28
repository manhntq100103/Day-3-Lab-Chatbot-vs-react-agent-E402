"""Tools cho ReAct Agent tư vấn quà tặng bằng LLM generation.

Role 2 chỉ định nghĩa tool, contract, validation và safeguards. Module không
hard-code API key và không phụ thuộc trực tiếp vào một SDK LLM cụ thể. Role 4
cấu hình một structured-LLM callable thông qua :func:`configure_tool_llm`.

Pipeline:
    1. ``extract_recipient_profile``: LLM trích xuất hồ sơ có cấu trúc.
    2. ``analyze_recipient_profile``: LLM tạo brief/insight chọn quà.
    3. ``generate_gift_candidates``: LLM sinh concept; Python validate, score,
       sort và gán rank.
    4. ``explain_recommendations``: LLM giải thích grounded; Python giữ khóa
       rank, ID, components và khoảng giá từ Tool 3.

Lưu ý:
    Candidate là ý tưởng quà do LLM sinh, không phải sản phẩm hoặc giá thị
    trường đã được xác minh. Mọi đề xuất đều cần kiểm tra lại trước khi mua.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeAlias

# Hỗ trợ hiển thị tiếng Việt trong Windows Console, không ảnh hưởng logic tool.
if getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

JsonObject: TypeAlias = dict[str, Any]
StructuredLLMResult: TypeAlias = JsonObject | str
StructuredLLMCallable: TypeAlias = Callable[
    [str, JsonObject, JsonObject], StructuredLLMResult
]

_TOOL_LLM: StructuredLLMCallable | None = None
_MAX_API_ATTEMPTS = 2  # Lần gọi đầu + tối đa 1 retry cho lỗi tạm thời.


@dataclass(frozen=True)
class _ToolFailure(Exception):
    """Lỗi nội bộ có thể chuyển an toàn thành Observation JSON."""

    code: str
    message: str
    field: str | None = None
    retryable: bool = False


# =============================================================================
# LLM CONFIGURATION
# =============================================================================

def configure_tool_llm(llm_callable: StructuredLLMCallable | None) -> None:
    """Cấu hình structured LLM callable được dùng bởi bốn public tool.

    Args:
        llm_callable: Hàm nhận ba tham số theo thứ tự ``system_prompt``,
            ``payload`` và ``response_schema``; trả về dictionary hoặc JSON
            string. Truyền ``None`` để gỡ cấu hình hiện tại.

    Returns:
        Không trả về dữ liệu.

    Error semantics:
        Ném ``TypeError`` nếu giá trị không callable và không phải ``None``.
        Đây là lỗi cấu hình của lập trình viên, không phải lỗi nghiệp vụ từ
        người dùng.

    Side effects:
        Thay đổi callable dùng chung trong module.

    Safety:
        Không nhận hoặc lưu API key. Role 4 chịu trách nhiệm tạo client và đọc
        API key từ biến môi trường.

    Example:
        >>> configure_tool_llm(my_structured_llm_adapter)
    """
    global _TOOL_LLM
    if llm_callable is not None and not callable(llm_callable):
        raise TypeError("llm_callable phải là callable hoặc None.")
    _TOOL_LLM = llm_callable


# =============================================================================
# COMMON HELPERS
# =============================================================================

def _json_response(data: Mapping[str, Any]) -> str:
    """Serialize một mapping thành JSON string giữ nguyên tiếng Việt."""
    return json.dumps(dict(data), ensure_ascii=False)


def _success_response(**payload: Any) -> str:
    """Tạo JSON success thống nhất cho Observation."""
    return _json_response({"ok": True, **payload})


def _error_response(
    code: str,
    message: str,
    *,
    field: str | None = None,
    retryable: bool = False,
) -> str:
    """Tạo JSON error an toàn, không làm ReAct loop crash."""
    error: JsonObject = {
        "code": code,
        "message": message,
        "retryable": retryable,
    }
    if field is not None:
        error["field"] = field
    return _json_response({"ok": False, "error": error})


def _normalize_text(value: Any) -> str:
    """Chuẩn hóa chuỗi để so sánh nhưng vẫn giữ dấu tiếng Việt."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().split())


def _slug_text(value: Any) -> str:
    """Chuẩn hóa chuỗi không dấu, dùng phát hiện tên concept gần trùng."""
    normalized = unicodedata.normalize("NFD", _normalize_text(value))
    ascii_text = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", ascii_text).strip()


def _unique_strings(value: Any, *, field: str, allow_empty: bool = True) -> list[str]:
    """Validate và chuẩn hóa list[str], loại phần tử rỗng/trùng."""
    if value is None and allow_empty:
        return []
    if not isinstance(value, list):
        raise _ToolFailure(
            "INVALID_SCHEMA",
            f"{field} phải là danh sách chuỗi.",
            field,
            True,
        )

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise _ToolFailure(
                "INVALID_SCHEMA",
                f"Mỗi phần tử của {field} phải là chuỗi.",
                field,
                True,
            )
        normalized = _normalize_text(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    if not allow_empty and not result:
        raise _ToolFailure(
            "INVALID_SCHEMA",
            f"{field} không được để trống.",
            field,
            True,
        )
    return result


def _parse_mapping(value: Any, *, field: str) -> JsonObject:
    """Nhận dictionary hoặc JSON string và trả dictionary độc lập."""
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise _ToolFailure("INVALID_INPUT", f"{field} không được rỗng.", field, True)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise _ToolFailure(
                "INVALID_JSON_INPUT",
                f"{field} không phải JSON hợp lệ.",
                field,
                True,
            ) from exc
        if isinstance(parsed, Mapping):
            return deepcopy(dict(parsed))
    raise _ToolFailure(
        "INVALID_INPUT",
        f"{field} phải là dictionary hoặc JSON object string.",
        field,
        True,
    )


def _parse_candidate_list(value: Any) -> list[JsonObject]:
    """Đọc danh sách candidate hoặc wrapper ``ranked_candidates``."""
    parsed: Any = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise _ToolFailure(
                "INVALID_JSON_INPUT",
                "gift_candidates không phải JSON hợp lệ.",
                "gift_candidates",
                True,
            ) from exc

    if isinstance(parsed, Mapping):
        parsed = parsed.get("ranked_candidates")
    if not isinstance(parsed, list) or not parsed:
        raise _ToolFailure(
            "EMPTY_GIFT_CANDIDATES",
            "gift_candidates phải là danh sách không rỗng.",
            "gift_candidates",
            True,
        )

    candidates: list[JsonObject] = []
    for index, candidate in enumerate(parsed):
        if not isinstance(candidate, Mapping):
            raise _ToolFailure(
                "INVALID_CANDIDATE_SCHEMA",
                f"Candidate tại vị trí {index} phải là object.",
                "gift_candidates",
                True,
            )
        candidates.append(deepcopy(dict(candidate)))
    return candidates


def _unwrap_object(data: JsonObject, key: str) -> JsonObject:
    """Lấy object con từ wrapper success nếu có."""
    nested = data.get(key)
    if isinstance(nested, Mapping):
        return deepcopy(dict(nested))
    data_field = data.get("data")
    if isinstance(data_field, Mapping):
        nested = data_field.get(key)
        if isinstance(nested, Mapping):
            return deepcopy(dict(nested))
    return deepcopy(data)


def _classify_provider_exception(exc: Exception) -> _ToolFailure:
    """Ánh xạ exception SDK phổ biến thành error code không lộ bí mật."""
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()

    if isinstance(exc, TimeoutError) or "timeout" in name or "timed out" in message:
        return _ToolFailure(
            "API_TIMEOUT",
            "Dịch vụ LLM phản hồi quá thời gian cho phép.",
            retryable=True,
        )
    if "ratelimit" in name or "rate limit" in message or "resourceexhausted" in name:
        return _ToolFailure(
            "API_RATE_LIMIT",
            "Dịch vụ LLM đang giới hạn tần suất yêu cầu.",
            retryable=True,
        )
    if (
        isinstance(exc, PermissionError)
        or "authentication" in name
        or "unauthorized" in message
        or "api key" in message
    ):
        return _ToolFailure(
            "API_AUTHENTICATION_ERROR",
            "Không thể xác thực dịch vụ LLM. Hãy kiểm tra cấu hình API ở Role 4.",
            retryable=False,
        )
    if isinstance(exc, ConnectionError) or "connection" in name:
        return _ToolFailure(
            "API_REQUEST_FAILED",
            "Không thể kết nối tới dịch vụ LLM.",
            retryable=True,
        )
    return _ToolFailure(
        "API_REQUEST_FAILED",
        "Dịch vụ LLM không hoàn thành yêu cầu.",
        retryable=False,
    )


def _call_structured_llm(
    *,
    system_prompt: str,
    payload: JsonObject,
    response_schema: JsonObject,
) -> JsonObject:
    """Gọi adapter LLM với tối đa một retry cho lỗi tạm thời."""
    if _TOOL_LLM is None:
        raise _ToolFailure(
            "TOOL_LLM_NOT_CONFIGURED",
            "Structured LLM cho tools chưa được cấu hình.",
            retryable=False,
        )

    last_failure: _ToolFailure | None = None
    for attempt in range(_MAX_API_ATTEMPTS):
        try:
            raw = _TOOL_LLM(system_prompt, deepcopy(payload), deepcopy(response_schema))
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise _ToolFailure(
                        "INVALID_JSON_RESPONSE",
                        "LLM không trả về JSON hợp lệ.",
                        retryable=True,
                    ) from exc
            if not isinstance(raw, Mapping):
                raise _ToolFailure(
                    "INVALID_LLM_RESPONSE",
                    "LLM không trả về object đúng định dạng.",
                    retryable=True,
                )
            return deepcopy(dict(raw))
        except _ToolFailure as failure:
            last_failure = failure
        except Exception as exc:  # Boundary với SDK bên ngoài.
            last_failure = _classify_provider_exception(exc)

        if last_failure is None or not last_failure.retryable:
            break
        if attempt + 1 >= _MAX_API_ATTEMPTS:
            break

    assert last_failure is not None
    raise last_failure


def _run_public_tool(operation: Callable[[], str]) -> str:
    """Boundary chung chuyển lỗi dự kiến thành JSON Observation."""
    try:
        return operation()
    except _ToolFailure as failure:
        return _error_response(
            failure.code,
            failure.message,
            field=failure.field,
            retryable=failure.retryable,
        )
    except Exception:
        # Không đưa exception nội bộ hoặc secret của provider ra Observation.
        return _error_response(
            "SYSTEM_ERROR",
            "Tool gặp lỗi nội bộ khi xử lý dữ liệu.",
            retryable=False,
        )


# =============================================================================
# PROFILE VALIDATION
# =============================================================================

def _validate_recipient_profile(value: Any) -> JsonObject:
    """Validate và chuẩn hóa recipient profile mà không mutate input."""
    wrapper = _parse_mapping(value, field="recipient_profile")
    profile = _unwrap_object(wrapper, "recipient_profile")

    traits = _unique_strings(profile.get("traits", []), field="traits")
    interests = _unique_strings(profile.get("interests", []), field="interests")
    preferences = _unique_strings(profile.get("preferences", []), field="preferences")
    exclusions = _unique_strings(profile.get("exclusions", []), field="exclusions")

    exclusion_set = set(exclusions)
    interests = [item for item in interests if item not in exclusion_set]
    preferences = [item for item in preferences if item not in exclusion_set]

    relationship = profile.get("relationship")
    if relationship is not None:
        if not isinstance(relationship, str):
            raise _ToolFailure(
                "INVALID_SCHEMA",
                "relationship phải là chuỗi hoặc null.",
                "relationship",
                True,
            )
        relationship = _normalize_text(relationship) or None

    occasion = profile.get("occasion")
    if occasion is not None:
        if not isinstance(occasion, str):
            raise _ToolFailure(
                "INVALID_SCHEMA",
                "occasion phải là chuỗi hoặc null.",
                "occasion",
                True,
            )
        occasion = _normalize_text(occasion) or None

    age = profile.get("age")
    if age is not None:
        if isinstance(age, bool) or not isinstance(age, int) or not 1 <= age <= 120:
            raise _ToolFailure(
                "INVALID_AGE",
                "age phải là số nguyên từ 1 đến 120 hoặc null.",
                "age",
                True,
            )

    budget = profile.get("budget_vnd")
    if budget is not None:
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise _ToolFailure(
                "INVALID_BUDGET",
                "budget_vnd phải là số nguyên dương hoặc null.",
                "budget_vnd",
                True,
            )

    return {
        "traits": traits,
        "interests": interests,
        "preferences": preferences,
        "exclusions": exclusions,
        "relationship": relationship,
        "occasion": occasion,
        "age": age,
        "budget_vnd": budget,
    }


def _validate_profile_analysis(value: Any, profile: JsonObject) -> JsonObject:
    """Validate analysis và khóa các constraint quan trọng bằng Python."""
    wrapper = _parse_mapping(value, field="profile_analysis")
    analysis = _unwrap_object(wrapper, "profile_analysis")

    priority_interests = _unique_strings(
        analysis.get("priority_interests", analysis.get("priority_tags", [])),
        field="priority_interests",
    )
    preferred_styles = _unique_strings(
        analysis.get("preferred_gift_styles", []),
        field="preferred_gift_styles",
    )
    avoid_features = _unique_strings(
        analysis.get("avoid_features", analysis.get("avoid_tags", [])),
        field="avoid_features",
    )
    avoid_features = list(dict.fromkeys([*profile["exclusions"], *avoid_features]))

    gift_goal = analysis.get("gift_goal", "")
    if not isinstance(gift_goal, str):
        raise _ToolFailure(
            "INVALID_SCHEMA", "gift_goal phải là chuỗi.", "gift_goal", True
        )
    gift_goal = gift_goal.strip()

    generation_guidelines = _unique_strings(
        analysis.get("generation_guidelines", []),
        field="generation_guidelines",
    )
    clarification_questions = _unique_strings(
        analysis.get("clarification_questions", []),
        field="clarification_questions",
    )
    analysis_notes = _unique_strings(
        analysis.get("analysis_notes", []), field="analysis_notes"
    )

    needs_clarification = analysis.get("needs_clarification", False)
    if not isinstance(needs_clarification, bool):
        raise _ToolFailure(
            "INVALID_SCHEMA",
            "needs_clarification phải là boolean.",
            "needs_clarification",
            True,
        )

    budget = profile["budget_vnd"]
    if budget is None:
        budget_strategy = {"minimum_vnd": None, "maximum_vnd": None}
        needs_clarification = True
        if "ngân sách tối đa cho món quà là bao nhiêu?" not in clarification_questions:
            clarification_questions.append("ngân sách tối đa cho món quà là bao nhiêu?")
    else:
        budget_strategy = {
            "minimum_vnd": int(budget * 0.55),
            "maximum_vnd": budget,
        }

    if not profile["interests"]:
        needs_clarification = True
        question = "người nhận có sở thích cụ thể nào không?"
        if question not in clarification_questions:
            clarification_questions.append(question)

    return {
        "priority_interests": priority_interests,
        "preferred_gift_styles": preferred_styles,
        "avoid_features": avoid_features,
        "gift_goal": gift_goal,
        "budget_strategy": budget_strategy,
        "generation_guidelines": generation_guidelines,
        "needs_clarification": needs_clarification,
        "clarification_questions": clarification_questions,
        "analysis_notes": analysis_notes,
    }


# =============================================================================
# TOOL PROMPTS AND SCHEMAS
# =============================================================================
_PROFILE_EXTRACTION_PROMPT = """
Bạn là tool trích xuất hồ sơ người nhận quà cho một ReAct Agent.
Chỉ lấy dữ kiện có trong mô tả; không chẩn đoán tâm lý và không tự điền thông
tin thiếu. Chuẩn hóa tuổi thành integer, ngân sách thành integer VND. Điều kiện
phủ định phải thắng sở thích tương ứng. Chỉ trả JSON đúng schema được cung cấp.
""".strip()

_PROFILE_ANALYSIS_PROMPT = """
Bạn là tool tạo brief chọn quà từ recipient_profile đã có cấu trúc. Tạo insight
có căn cứ, không chẩn đoán tâm lý, không sinh sản phẩm và không thêm sở thích
không được hỗ trợ bởi hồ sơ. Exclusions phải xuất hiện trong avoid_features.
Chỉ trả JSON đúng schema được cung cấp.
""".strip()

_GIFT_GENERATION_PROMPT = """
Bạn là tool sáng tạo concept quà tặng. Hãy sinh các concept khác nhau đáng kể,
có thể là món đơn hoặc gift bundle, dựa đúng recipient_profile và
profile_analysis. Không dùng catalog cố định, không tạo tên shop, thương hiệu,
URL hoặc tuyên bố tồn kho. Giá chỉ là ước tính. Không tạo concept vượt ngân sách
hoặc vi phạm exclusions/avoid_features. Không tự gán score, rank hay candidate_id;
Python sẽ validate và thực hiện các bước đó. Chỉ trả JSON đúng schema.
""".strip()

_EXPLANATION_PROMPT = """
Bạn là tool giải thích các concept quà đã được Python khóa và xếp hạng. Chỉ tạo
reason, why_it_fits, personalization_tip, verify_before_buying và budget_note.
Không đổi candidate_id, rank, name, components, khoảng giá hoặc score; không
thêm candidate, sản phẩm, thương hiệu hay URL. Chỉ trả JSON đúng schema.
""".strip()

PROFILE_EXTRACTION_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {
        "traits": {"type": "array", "items": {"type": "string"}},
        "interests": {"type": "array", "items": {"type": "string"}},
        "preferences": {"type": "array", "items": {"type": "string"}},
        "exclusions": {"type": "array", "items": {"type": "string"}},
        "relationship": {"type": ["string", "null"]},
        "occasion": {"type": ["string", "null"]},
        "age": {"type": ["integer", "null"]},
        "budget_vnd": {"type": ["integer", "null"]},
    },
    "required": [
        "traits",
        "interests",
        "preferences",
        "exclusions",
        "relationship",
        "occasion",
        "age",
        "budget_vnd",
    ],
    "additionalProperties": False,
}

PROFILE_ANALYSIS_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {
        "priority_interests": {"type": "array", "items": {"type": "string"}},
        "preferred_gift_styles": {"type": "array", "items": {"type": "string"}},
        "avoid_features": {"type": "array", "items": {"type": "string"}},
        "gift_goal": {"type": "string"},
        "generation_guidelines": {"type": "array", "items": {"type": "string"}},
        "needs_clarification": {"type": "boolean"},
        "clarification_questions": {"type": "array", "items": {"type": "string"}},
        "analysis_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "priority_interests",
        "preferred_gift_styles",
        "avoid_features",
        "gift_goal",
        "generation_guidelines",
        "needs_clarification",
        "clarification_questions",
        "analysis_notes",
    ],
    "additionalProperties": False,
}

_GENERATED_CANDIDATE_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "concept": {"type": "string"},
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "estimated_price_vnd": {"type": "integer"},
                },
                "required": ["name", "estimated_price_vnd"],
                "additionalProperties": False,
            },
        },
        "estimated_price_range_vnd": {
            "type": "object",
            "properties": {
                "minimum": {"type": "integer"},
                "maximum": {"type": "integer"},
            },
            "required": ["minimum", "maximum"],
            "additionalProperties": False,
        },
        "fit_tags": {"type": "array", "items": {"type": "string"}},
        "gift_styles": {"type": "array", "items": {"type": "string"}},
        "suitable_occasions": {"type": "array", "items": {"type": "string"}},
        "suitable_relationships": {"type": "array", "items": {"type": "string"}},
        "possible_risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "name",
        "concept",
        "components",
        "estimated_price_range_vnd",
        "fit_tags",
        "gift_styles",
        "suitable_occasions",
        "suitable_relationships",
        "possible_risks",
    ],
    "additionalProperties": False,
}

GIFT_GENERATION_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": _GENERATED_CANDIDATE_SCHEMA,
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

EXPLANATION_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {
        "explanations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "why_it_fits": {"type": "array", "items": {"type": "string"}},
                    "personalization_tip": {"type": "string"},
                    "verify_before_buying": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "budget_note": {"type": "string"},
                },
                "required": [
                    "candidate_id",
                    "reason",
                    "why_it_fits",
                    "personalization_tip",
                    "verify_before_buying",
                    "budget_note",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["explanations"],
    "additionalProperties": False,
}


# =============================================================================
# TOOL 1
# =============================================================================

def extract_recipient_profile(user_description: str) -> str:
    """Trích xuất mô tả tự do thành recipient profile có cấu trúc.

    Role in pipeline:
        Tool 1/4, gọi đầu tiên.

    Args:
        user_description: Mô tả người nhận, dịp tặng và ngân sách.

    Returns:
        JSON string chứa ``recipient_profile``, ``missing_fields`` và nguồn dữ
        liệu; hoặc structured error.

    Error semantics:
        Trả JSON error khi input rỗng, LLM chưa cấu hình, API lỗi hoặc structured
        output không hợp lệ. Không làm ReAct loop crash.

    Use when:
        Người dùng cung cấp mô tả bằng ngôn ngữ tự nhiên.

    Do not use when:
        Đã có recipient profile hợp lệ hoặc cần sinh quà.

    Side effects:
        Gọi LLM thông qua adapter đã cấu hình.

    Safety:
        Python validate lại output; exclusion thắng interest và không tự đoán
        dữ kiện còn thiếu.
    """

    def operation() -> str:
        if not isinstance(user_description, str) or not user_description.strip():
            raise _ToolFailure(
                "INVALID_INPUT",
                "user_description phải là chuỗi không rỗng.",
                "user_description",
                True,
            )

        raw_profile = _call_structured_llm(
            system_prompt=_PROFILE_EXTRACTION_PROMPT,
            payload={"user_description": user_description.strip()},
            response_schema=PROFILE_EXTRACTION_SCHEMA,
        )
        profile = _validate_recipient_profile(raw_profile)

        missing_fields: list[str] = []
        if not profile["interests"]:
            missing_fields.append("interests")
        if profile["budget_vnd"] is None:
            missing_fields.append("budget_vnd")
        if profile["relationship"] is None:
            missing_fields.append("relationship")
        if profile["occasion"] is None:
            missing_fields.append("occasion")

        return _success_response(
            recipient_profile=profile,
            missing_fields=missing_fields,
            source="llm_structured_extraction",
        )

    return _run_public_tool(operation)


# =============================================================================
# TOOL 2
# =============================================================================

def analyze_recipient_profile(recipient_profile: JsonObject | str) -> str:
    """Tạo brief/insight chọn quà từ recipient profile.

    Role in pipeline:
        Tool 2/4, chỉ gọi sau Observation của Tool 1.

    Args:
        recipient_profile: Profile object hoặc JSON string từ Tool 1.

    Returns:
        JSON string chứa ``profile_analysis`` hoặc structured error.

    Error semantics:
        Validation Python từ chối profile sai schema hoặc ngân sách không hợp
        lệ; lỗi API được chuyển thành Observation an toàn.

    Use when:
        Cần tạo chiến lược và guideline để sinh concept.

    Do not use when:
        Chưa có profile hoặc cần sản phẩm cụ thể.

    Side effects:
        Gọi LLM qua adapter.

    Safety:
        Không chẩn đoán tâm lý; Python buộc exclusions vào avoid_features và
        tự khóa budget strategy.
    """

    def operation() -> str:
        profile = _validate_recipient_profile(recipient_profile)
        raw_analysis = _call_structured_llm(
            system_prompt=_PROFILE_ANALYSIS_PROMPT,
            payload={"recipient_profile": profile},
            response_schema=PROFILE_ANALYSIS_SCHEMA,
        )
        analysis = _validate_profile_analysis(raw_analysis, profile)
        return _success_response(
            profile_analysis=analysis,
            source="llm_structured_analysis",
        )

    return _run_public_tool(operation)


# =============================================================================
# TOOL 3 HELPERS
# =============================================================================

def _validate_component(value: Any, *, index: int) -> JsonObject:
    """Validate một component trong concept quà."""
    if not isinstance(value, Mapping):
        raise _ToolFailure(
            "INVALID_CANDIDATE_SCHEMA",
            f"Component {index} phải là object.",
            "components",
            True,
        )
    name = value.get("name")
    price = value.get("estimated_price_vnd")
    if not isinstance(name, str) or not name.strip():
        raise _ToolFailure(
            "INVALID_CANDIDATE_SCHEMA",
            f"Component {index} thiếu name hợp lệ.",
            "components",
            True,
        )
    if isinstance(price, bool) or not isinstance(price, int) or price < 0:
        raise _ToolFailure(
            "INVALID_CANDIDATE_SCHEMA",
            f"Component {index} có estimated_price_vnd không hợp lệ.",
            "components",
            True,
        )
    return {"name": name.strip(), "estimated_price_vnd": price}


def _validate_generated_candidate(value: Any) -> JsonObject:
    """Validate schema một concept do LLM sinh."""
    if not isinstance(value, Mapping):
        raise _ToolFailure(
            "INVALID_CANDIDATE_SCHEMA",
            "Candidate phải là object.",
            retryable=True,
        )

    name = value.get("name")
    concept = value.get("concept")
    if not isinstance(name, str) or not name.strip():
        raise _ToolFailure(
            "INVALID_CANDIDATE_SCHEMA", "Candidate thiếu name.", "name", True
        )
    if not isinstance(concept, str) or not concept.strip():
        raise _ToolFailure(
            "INVALID_CANDIDATE_SCHEMA", "Candidate thiếu concept.", "concept", True
        )

    components_raw = value.get("components")
    if not isinstance(components_raw, list) or not components_raw:
        raise _ToolFailure(
            "INVALID_CANDIDATE_SCHEMA",
            "components phải là danh sách không rỗng.",
            "components",
            True,
        )
    components = [
        _validate_component(component, index=index)
        for index, component in enumerate(components_raw, start=1)
    ]

    price_range = value.get("estimated_price_range_vnd")
    if not isinstance(price_range, Mapping):
        raise _ToolFailure(
            "INVALID_CANDIDATE_SCHEMA",
            "estimated_price_range_vnd phải là object.",
            "estimated_price_range_vnd",
            True,
        )
    minimum = price_range.get("minimum")
    maximum = price_range.get("maximum")
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, int)
        or not isinstance(maximum, int)
        or minimum < 0
        or maximum <= 0
        or minimum > maximum
    ):
        raise _ToolFailure(
            "INVALID_CANDIDATE_SCHEMA",
            "Khoảng giá candidate không hợp lệ.",
            "estimated_price_range_vnd",
            True,
        )

    if re.search(r"https?://|www\.", f"{name} {concept}", flags=re.IGNORECASE):
        raise _ToolFailure(
            "INVALID_CANDIDATE_SCHEMA",
            "Candidate không được chứa URL.",
            "concept",
            True,
        )

    return {
        "name": name.strip(),
        "concept": concept.strip(),
        "components": components,
        "estimated_price_range_vnd": {"minimum": minimum, "maximum": maximum},
        "fit_tags": _unique_strings(value.get("fit_tags", []), field="fit_tags"),
        "gift_styles": _unique_strings(
            value.get("gift_styles", []), field="gift_styles"
        ),
        "suitable_occasions": _unique_strings(
            value.get("suitable_occasions", []), field="suitable_occasions"
        ),
        "suitable_relationships": _unique_strings(
            value.get("suitable_relationships", []), field="suitable_relationships"
        ),
        "possible_risks": _unique_strings(
            value.get("possible_risks", []), field="possible_risks"
        ),
    }


def _candidate_violation(candidate: JsonObject, avoid_features: Sequence[str]) -> str | None:
    """Trả feature vi phạm đầu tiên hoặc None."""
    if not avoid_features:
        return None
    searchable_parts = [
        candidate["name"],
        candidate["concept"],
        *candidate["fit_tags"],
        *candidate["gift_styles"],
        *(component["name"] for component in candidate["components"]),
    ]
    searchable = _normalize_text(" ".join(searchable_parts))
    for feature in avoid_features:
        normalized = _normalize_text(feature)
        if normalized and normalized in searchable:
            return feature
    return None


def _score_candidate(
    candidate: JsonObject,
    profile: JsonObject,
    analysis: JsonObject,
) -> tuple[int, JsonObject, list[str]]:
    """Tính điểm deterministic cho một concept hợp lệ."""
    fit_tags = set(candidate["fit_tags"])
    styles = set(candidate["gift_styles"])
    interests = set(profile["interests"])
    preferences = set(profile["preferences"])
    preferred_styles = set(analysis["preferred_gift_styles"])

    matched_interests = sorted(fit_tags & interests)
    matched_styles = sorted(styles & preferred_styles)
    matched_preferences = sorted((fit_tags | styles) & preferences)

    interest_points = len(matched_interests) * 5
    style_points = len(matched_styles) * 3
    preference_points = len(matched_preferences) * 3
    personalization_points = 3 if "cá nhân hóa" in styles or "cá nhân hóa" in fit_tags else 0

    occasion = profile.get("occasion")
    occasion_points = (
        2 if occasion and occasion in set(candidate["suitable_occasions"]) else 0
    )
    relationship = profile.get("relationship")
    relationship_points = (
        2
        if relationship and relationship in set(candidate["suitable_relationships"])
        else 0
    )

    maximum = candidate["estimated_price_range_vnd"]["maximum"]
    budget = profile["budget_vnd"]
    budget_points = 2 if isinstance(budget, int) and maximum <= budget else 0

    strategy = analysis["budget_strategy"]
    strategy_min = strategy.get("minimum_vnd")
    strategy_max = strategy.get("maximum_vnd")
    budget_strategy_points = (
        2
        if isinstance(strategy_min, int)
        and isinstance(strategy_max, int)
        and strategy_min <= maximum <= strategy_max
        else 0
    )

    risk_penalty = -len(candidate["possible_risks"])
    score = (
        interest_points
        + style_points
        + preference_points
        + personalization_points
        + occasion_points
        + relationship_points
        + budget_points
        + budget_strategy_points
        + risk_penalty
    )

    breakdown = {
        "interest_points": interest_points,
        "preferred_style_points": style_points,
        "preference_points": preference_points,
        "personalization_points": personalization_points,
        "occasion_points": occasion_points,
        "relationship_points": relationship_points,
        "budget_points": budget_points,
        "budget_strategy_points": budget_strategy_points,
        "risk_penalty": risk_penalty,
    }
    matched_signals = list(
        dict.fromkeys([*matched_interests, *matched_styles, *matched_preferences])
    )
    return score, breakdown, matched_signals


# =============================================================================
# TOOL 3
# =============================================================================

def generate_gift_candidates(
    recipient_profile: JsonObject | str,
    profile_analysis: JsonObject | str,
    max_candidates: int = 10,
) -> str:
    """Sinh concept quà bằng LLM, sau đó Python validate, score và rank.

    Role in pipeline:
        Tool 3/4, gọi sau Tool 1 và Tool 2. Tool này đã thực hiện ranking.

    Args:
        recipient_profile: Hồ sơ từ Tool 1.
        profile_analysis: Brief từ Tool 2.
        max_candidates: Số concept tối đa trả về, từ 1 đến 20.

    Returns:
        JSON string chứa ``ranked_candidates`` và ``generation_summary``.

    Error semantics:
        Trả structured error nếu thiếu ngân sách, API lỗi hoặc không còn concept
        hợp lệ sau validation.

    Use when:
        Cần sinh các ý tưởng quà mới phù hợp hồ sơ.

    Do not use when:
        Chưa có profile/analysis hoặc muốn tìm sản phẩm và tồn kho thực tế.

    Side effects:
        Gọi LLM; không ghi file và không thay đổi input.

    Safety:
        Không dùng catalog cố định. Python loại concept vượt ngân sách, vi phạm
        exclusion, sai schema, chứa URL hoặc trùng tên; Python tự score/rank.
    """

    def operation() -> str:
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or not 1 <= max_candidates <= 20
        ):
            raise _ToolFailure(
                "INVALID_MAX_CANDIDATES",
                "max_candidates phải là số nguyên từ 1 đến 20.",
                "max_candidates",
                True,
            )

        profile = _validate_recipient_profile(recipient_profile)
        analysis = _validate_profile_analysis(profile_analysis, profile)
        budget = profile["budget_vnd"]
        if budget is None:
            raise _ToolFailure(
                "MISSING_BUDGET",
                "Cần ngân sách trước khi sinh concept quà.",
                "budget_vnd",
                True,
            )

        requested_generation_count = min(20, max_candidates + 3)
        raw_generation = _call_structured_llm(
            system_prompt=_GIFT_GENERATION_PROMPT,
            payload={
                "recipient_profile": profile,
                "profile_analysis": analysis,
                "number_of_candidates": requested_generation_count,
            },
            response_schema=GIFT_GENERATION_SCHEMA,
        )
        raw_candidates = raw_generation.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise _ToolFailure(
                "EMPTY_GENERATION",
                "LLM không sinh được candidate nào.",
                retryable=True,
            )

        valid_candidates: list[JsonObject] = []
        rejected: list[JsonObject] = []
        seen_names: set[str] = set()
        avoid_features = list(
            dict.fromkeys([*profile["exclusions"], *analysis["avoid_features"]])
        )

        for index, raw_candidate in enumerate(raw_candidates, start=1):
            fallback_name = (
                raw_candidate.get("name", f"candidate_{index}")
                if isinstance(raw_candidate, Mapping)
                else f"candidate_{index}"
            )
            try:
                candidate = _validate_generated_candidate(raw_candidate)
            except _ToolFailure as failure:
                rejected.append(
                    {
                        "candidate_name": fallback_name,
                        "reason_code": failure.code,
                        "reason": failure.message,
                    }
                )
                continue

            name_key = _slug_text(candidate["name"])
            if not name_key or name_key in seen_names:
                rejected.append(
                    {
                        "candidate_name": candidate["name"],
                        "reason_code": "DUPLICATE_CANDIDATE",
                        "reason": "Tên concept bị trùng hoặc quá giống concept trước đó.",
                    }
                )
                continue
            seen_names.add(name_key)

            maximum = candidate["estimated_price_range_vnd"]["maximum"]
            if maximum > budget:
                rejected.append(
                    {
                        "candidate_name": candidate["name"],
                        "reason_code": "CANDIDATE_OVER_BUDGET",
                        "reason": "Khoảng giá tối đa vượt ngân sách.",
                    }
                )
                continue

            violation = _candidate_violation(candidate, avoid_features)
            if violation is not None:
                rejected.append(
                    {
                        "candidate_name": candidate["name"],
                        "reason_code": "CANDIDATE_VIOLATES_EXCLUSION",
                        "reason": f"Concept vi phạm điều cần tránh: {violation}.",
                    }
                )
                continue

            candidate_id = f"C{index:03d}"
            score, breakdown, matched_signals = _score_candidate(
                candidate, profile, analysis
            )
            valid_candidates.append(
                {
                    **candidate,
                    "candidate_id": candidate_id,
                    "score": score,
                    "score_breakdown": breakdown,
                    "matched_signals": matched_signals,
                    "data_source": "llm_generated_concept",
                    "requires_market_verification": True,
                }
            )

        if not valid_candidates:
            raise _ToolFailure(
                "NO_VALID_CANDIDATES",
                "Không còn concept hợp lệ sau kiểm tra ngân sách, exclusion và schema.",
                retryable=True,
            )

        valid_candidates.sort(
            key=lambda item: (
                -item["score"],
                item["estimated_price_range_vnd"]["maximum"],
                item["candidate_id"],
            )
        )

        ranked: list[JsonObject] = []
        for rank, candidate in enumerate(valid_candidates[:max_candidates], start=1):
            ranked.append({"rank": rank, **candidate})

        return _success_response(
            ranked_candidates=ranked,
            generation_summary={
                "requested_count": max_candidates,
                "generation_requested_from_llm": requested_generation_count,
                "generated_count": len(raw_candidates),
                "valid_count": len(valid_candidates),
                "returned_count": len(ranked),
                "rejected_candidates": rejected,
            },
            generation_note=(
                "Các candidate là concept do LLM sinh; giá chỉ là ước tính và "
                "cần xác minh thị trường."
            ),
        )

    return _run_public_tool(operation)


# =============================================================================
# TOOL 4
# =============================================================================

def _validate_locked_candidate(candidate: JsonObject) -> JsonObject:
    """Validate các trường cốt lõi mà Tool 4 không được thay đổi."""
    candidate_id = candidate.get("candidate_id")
    rank = candidate.get("rank")
    name = candidate.get("name")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise _ToolFailure(
            "INVALID_CANDIDATE_SCHEMA",
            "Candidate thiếu candidate_id.",
            "candidate_id",
            True,
        )
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise _ToolFailure(
            "INVALID_CANDIDATE_SCHEMA", "Candidate có rank không hợp lệ.", "rank", True
        )
    if not isinstance(name, str) or not name.strip():
        raise _ToolFailure(
            "INVALID_CANDIDATE_SCHEMA", "Candidate thiếu name.", "name", True
        )

    validated_generated = _validate_generated_candidate(candidate)
    score = candidate.get("score")
    if isinstance(score, bool) or not isinstance(score, int):
        raise _ToolFailure(
            "INVALID_CANDIDATE_SCHEMA", "Candidate thiếu score hợp lệ.", "score", True
        )
    score_breakdown = candidate.get("score_breakdown")
    if not isinstance(score_breakdown, Mapping):
        raise _ToolFailure(
            "INVALID_CANDIDATE_SCHEMA",
            "Candidate thiếu score_breakdown.",
            "score_breakdown",
            True,
        )

    return {
        "rank": rank,
        "candidate_id": candidate_id.strip(),
        **validated_generated,
        "score": score,
        "score_breakdown": deepcopy(dict(score_breakdown)),
        "matched_signals": _unique_strings(
            candidate.get("matched_signals", []), field="matched_signals"
        ),
        "data_source": "llm_generated_concept",
        "requires_market_verification": True,
    }


def _validate_explanation(value: Any) -> JsonObject:
    """Validate phần giải thích được phép do LLM tạo."""
    if not isinstance(value, Mapping):
        raise _ToolFailure(
            "INVALID_LLM_RESPONSE",
            "Mỗi explanation phải là object.",
            retryable=True,
        )
    candidate_id = value.get("candidate_id")
    reason = value.get("reason")
    personalization_tip = value.get("personalization_tip")
    budget_note = value.get("budget_note")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise _ToolFailure(
            "INVALID_LLM_RESPONSE", "Explanation thiếu candidate_id.", retryable=True
        )
    for field_name, field_value in (
        ("reason", reason),
        ("personalization_tip", personalization_tip),
        ("budget_note", budget_note),
    ):
        if not isinstance(field_value, str) or not field_value.strip():
            raise _ToolFailure(
                "INVALID_LLM_RESPONSE",
                f"Explanation thiếu {field_name} hợp lệ.",
                field_name,
                True,
            )
    return {
        "candidate_id": candidate_id.strip(),
        "reason": reason.strip(),
        "why_it_fits": _unique_strings(
            value.get("why_it_fits", []), field="why_it_fits"
        ),
        "personalization_tip": personalization_tip.strip(),
        "verify_before_buying": _unique_strings(
            value.get("verify_before_buying", []), field="verify_before_buying"
        ),
        "budget_note": budget_note.strip(),
    }


def explain_recommendations(
    recipient_profile: JsonObject | str,
    profile_analysis: JsonObject | str,
    gift_candidates: list[JsonObject] | str | JsonObject,
    top_k: int = 5,
) -> str:
    """Giải thích grounded cho candidate đã được Tool 3 xếp hạng.

    Role in pipeline:
        Tool 4/4, bước cuối trước Final Answer.

    Args:
        recipient_profile: Profile từ Tool 1.
        profile_analysis: Brief từ Tool 2.
        gift_candidates: ``ranked_candidates`` từ Tool 3 hoặc wrapper chứa nó.
        top_k: Số candidate cần giải thích, từ 1 đến 20.

    Returns:
        JSON string chứa recommendations đã khóa dữ liệu cốt lõi.

    Error semantics:
        Trả structured error khi candidate/schema/top_k hoặc LLM response không
        hợp lệ.

    Use when:
        Đã có candidate được Python score và rank.

    Do not use when:
        Muốn đổi ranking, components, khoảng giá hoặc sinh concept mới.

    Side effects:
        Gọi LLM qua adapter.

    Safety:
        Python merge explanation vào candidate gốc; mọi thay đổi rank, ID, giá,
        components hoặc score do LLM đề xuất đều bị bỏ qua.
    """

    def operation() -> str:
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 20
        ):
            raise _ToolFailure(
                "INVALID_TOP_K",
                "top_k phải là số nguyên từ 1 đến 20.",
                "top_k",
                True,
            )

        profile = _validate_recipient_profile(recipient_profile)
        analysis = _validate_profile_analysis(profile_analysis, profile)
        raw_candidates = _parse_candidate_list(gift_candidates)
        locked_candidates = [
            _validate_locked_candidate(candidate) for candidate in raw_candidates[:top_k]
        ]

        seen_ids: set[str] = set()
        for candidate in locked_candidates:
            candidate_id = candidate["candidate_id"]
            if candidate_id in seen_ids:
                raise _ToolFailure(
                    "DUPLICATE_CANDIDATE",
                    f"candidate_id bị trùng: {candidate_id}.",
                    "candidate_id",
                    True,
                )
            seen_ids.add(candidate_id)

        raw_explanations = _call_structured_llm(
            system_prompt=_EXPLANATION_PROMPT,
            payload={
                "recipient_profile": profile,
                "profile_analysis": analysis,
                "locked_candidates": locked_candidates,
            },
            response_schema=EXPLANATION_SCHEMA,
        )
        explanations_raw = raw_explanations.get("explanations")
        if not isinstance(explanations_raw, list) or not explanations_raw:
            raise _ToolFailure(
                "INVALID_LLM_RESPONSE",
                "LLM không trả về danh sách explanations.",
                retryable=True,
            )

        explanation_by_id: dict[str, JsonObject] = {}
        for raw in explanations_raw:
            explanation = _validate_explanation(raw)
            candidate_id = explanation["candidate_id"]
            if candidate_id in explanation_by_id:
                raise _ToolFailure(
                    "INVALID_LLM_RESPONSE",
                    f"Explanation bị trùng candidate_id: {candidate_id}.",
                    retryable=True,
                )
            explanation_by_id[candidate_id] = explanation

        recommendations: list[JsonObject] = []
        for candidate in locked_candidates:
            candidate_id = candidate["candidate_id"]
            explanation = explanation_by_id.get(candidate_id)
            if explanation is None:
                raise _ToolFailure(
                    "INVALID_LLM_RESPONSE",
                    f"Thiếu explanation cho {candidate_id}.",
                    retryable=True,
                )
            recommendations.append(
                {
                    # Các trường khóa lấy từ Tool 3.
                    "rank": candidate["rank"],
                    "candidate_id": candidate_id,
                    "name": candidate["name"],
                    "concept": candidate["concept"],
                    "components": candidate["components"],
                    "estimated_price_range_vnd": candidate[
                        "estimated_price_range_vnd"
                    ],
                    "score": candidate["score"],
                    "matched_signals": candidate["matched_signals"],
                    # Chỉ các trường dưới lấy từ LLM Tool 4.
                    "reason": explanation["reason"],
                    "why_it_fits": explanation["why_it_fits"],
                    "personalization_tip": explanation["personalization_tip"],
                    "verify_before_buying": explanation["verify_before_buying"],
                    "budget_note": explanation["budget_note"],
                    "data_source": "llm_generated_concept",
                    "requires_market_verification": True,
                }
            )

        return _success_response(
            recommendations=recommendations,
            explanation_note=(
                "Thứ tự, ID, components, score và khoảng giá được giữ nguyên "
                "từ generate_gift_candidates."
            ),
        )

    return _run_public_tool(operation)


# =============================================================================
# REGISTRY AND SPECS
# =============================================================================
AVAILABLE_TOOLS: dict[str, Callable[..., str]] = {
    "extract_recipient_profile": extract_recipient_profile,
    "analyze_recipient_profile": analyze_recipient_profile,
    "generate_gift_candidates": generate_gift_candidates,
    "explain_recommendations": explain_recommendations,
}

TOOL_CONTRACTS: dict[str, JsonObject] = {
    "extract_recipient_profile": {
        "step": 1,
        "input": {"user_description": "str"},
        "output": {"recipient_profile": "object", "missing_fields": "list[str]"},
        "llm_role": "structured extraction",
    },
    "analyze_recipient_profile": {
        "step": 2,
        "input": {"recipient_profile": "object"},
        "output": {"profile_analysis": "object"},
        "llm_role": "structured analysis",
    },
    "generate_gift_candidates": {
        "step": 3,
        "input": {
            "recipient_profile": "object",
            "profile_analysis": "object",
            "max_candidates": "int",
        },
        "output": {"ranked_candidates": "list", "generation_summary": "object"},
        "llm_role": "creative concept generation",
        "python_role": "validation, filtering, scoring and ranking",
    },
    "explain_recommendations": {
        "step": 4,
        "input": {
            "recipient_profile": "object",
            "profile_analysis": "object",
            "gift_candidates": "list",
            "top_k": "int",
        },
        "output": {"recommendations": "list"},
        "llm_role": "grounded explanation only",
    },
}

TOOL_SPECS: list[JsonObject] = [
    {
        "name": "extract_recipient_profile",
        "description": (
            "Tool 1/4. Gọi đầu tiên để dùng structured LLM trích xuất hồ sơ "
            "từ mô tả tự nhiên. Không sinh quà và không tự đoán dữ kiện thiếu."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_description": {
                    "type": "string",
                    "description": "Mô tả tự nhiên về người nhận và yêu cầu tặng quà.",
                }
            },
            "required": ["user_description"],
            "additionalProperties": False,
        },
    },
    {
        "name": "analyze_recipient_profile",
        "description": (
            "Tool 2/4. Chỉ gọi sau Tool 1 và truyền recipient_profile từ "
            "Observation trước. Tạo brief sinh quà, không tạo candidate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "recipient_profile": {
                    "type": "object",
                    "description": "recipient_profile từ Tool 1.",
                }
            },
            "required": ["recipient_profile"],
            "additionalProperties": False,
        },
    },
    {
        "name": "generate_gift_candidates",
        "description": (
            "Tool 3/4. Dùng LLM sinh concept quà mới, không dùng catalog cố "
            "định. Python kiểm tra schema/ngân sách/exclusions, tự gán ID, "
            "score và rank. Giá là ước tính, không phải dữ liệu thị trường."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "recipient_profile": {
                    "type": "object",
                    "description": "recipient_profile từ Tool 1.",
                },
                "profile_analysis": {
                    "type": "object",
                    "description": "profile_analysis từ Tool 2.",
                },
                "max_candidates": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 10,
                },
            },
            "required": ["recipient_profile", "profile_analysis"],
            "additionalProperties": False,
        },
    },
    {
        "name": "explain_recommendations",
        "description": (
            "Tool 4/4. Chỉ giải thích ranked_candidates từ Tool 3. Python "
            "khóa rank, candidate_id, components, score và khoảng giá; LLM "
            "không được sửa hoặc thêm candidate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "recipient_profile": {"type": "object"},
                "profile_analysis": {"type": "object"},
                "gift_candidates": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                },
            },
            "required": ["recipient_profile", "profile_analysis", "gift_candidates"],
            "additionalProperties": False,
        },
    },
]


# =============================================================================
# DETERMINISTIC SMOKE TESTS (NO REAL API)
# =============================================================================

def _fake_structured_llm(
    system_prompt: str,
    payload: JsonObject,
    response_schema: JsonObject,
) -> JsonObject:
    """Fake LLM cho test; không dùng mạng hoặc API token."""
    del response_schema
    if "trích xuất hồ sơ" in system_prompt:
        text = _normalize_text(payload.get("user_description", ""))
        exclusions = ["mùi hương"] if "không thích mùi hương" in text else []
        interests = ["đọc sách", "trà"]
        if "không uống trà" in text:
            exclusions.append("trà")
            interests = [item for item in interests if item != "trà"]
        return {
            "traits": ["hướng nội"],
            "interests": interests,
            "preferences": ["ý nghĩa", "cá nhân hóa"],
            "exclusions": exclusions,
            "relationship": "bạn thân",
            "occasion": "sinh nhật",
            "age": 21,
            "budget_vnd": 800_000 if "không ngân sách" not in text else None,
        }

    if "tạo brief chọn quà" in system_prompt:
        profile = payload["recipient_profile"]
        return {
            "priority_interests": profile["interests"],
            "preferred_gift_styles": ["ý nghĩa", "cá nhân hóa", "thư giãn"],
            "avoid_features": profile["exclusions"],
            "gift_goal": "Tạo cảm giác được thấu hiểu qua sở thích cá nhân.",
            "generation_guidelines": ["Kết hợp sở thích nổi bật."],
            "needs_clarification": False,
            "clarification_questions": [],
            "analysis_notes": [],
        }

    if "sáng tạo concept quà tặng" in system_prompt:
        return {
            "candidates": [
                {
                    "name": "Bộ đọc sách và thưởng trà cá nhân hóa",
                    "concept": "Kết hợp sách, trà, bookmark khắc tên và thiệp.",
                    "components": [
                        {"name": "Sách theo thể loại yêu thích", "estimated_price_vnd": 250000},
                        {"name": "Hộp trà tuyển chọn", "estimated_price_vnd": 220000},
                        {"name": "Bookmark cá nhân hóa", "estimated_price_vnd": 80000},
                    ],
                    "estimated_price_range_vnd": {"minimum": 500000, "maximum": 700000},
                    "fit_tags": ["đọc sách", "trà", "cá nhân hóa", "ý nghĩa"],
                    "gift_styles": ["cá nhân hóa", "ý nghĩa", "thư giãn"],
                    "suitable_occasions": ["sinh nhật"],
                    "suitable_relationships": ["bạn thân"],
                    "possible_risks": ["Cần biết thể loại sách yêu thích."],
                },
                {
                    "name": "Góc thư giãn đọc sách tối giản",
                    "concept": "Đèn đọc sách, gối tựa và sổ ghi chú.",
                    "components": [
                        {"name": "Đèn đọc sách", "estimated_price_vnd": 300000},
                        {"name": "Sổ ghi chú", "estimated_price_vnd": 90000},
                    ],
                    "estimated_price_range_vnd": {"minimum": 390000, "maximum": 520000},
                    "fit_tags": ["đọc sách", "thư giãn"],
                    "gift_styles": ["thư giãn", "thực dụng"],
                    "suitable_occasions": ["sinh nhật"],
                    "suitable_relationships": ["bạn thân"],
                    "possible_risks": [],
                },
                {
                    "name": "Concept vượt ngân sách",
                    "concept": "Một concept test phải bị Python loại.",
                    "components": [{"name": "Thiết bị đắt tiền", "estimated_price_vnd": 1200000}],
                    "estimated_price_range_vnd": {"minimum": 1000000, "maximum": 1200000},
                    "fit_tags": ["công nghệ"],
                    "gift_styles": ["thực dụng"],
                    "suitable_occasions": ["sinh nhật"],
                    "suitable_relationships": ["bạn thân"],
                    "possible_risks": [],
                },
                {
                    "name": "Bộ nến mùi hương",
                    "concept": "Concept phải bị loại nếu profile tránh mùi hương.",
                    "components": [{"name": "Nến mùi hương", "estimated_price_vnd": 250000}],
                    "estimated_price_range_vnd": {"minimum": 200000, "maximum": 300000},
                    "fit_tags": ["mùi hương", "thư giãn"],
                    "gift_styles": ["thư giãn"],
                    "suitable_occasions": ["sinh nhật"],
                    "suitable_relationships": ["bạn thân"],
                    "possible_risks": [],
                },
            ]
        }

    if "giải thích các concept" in system_prompt:
        explanations = []
        for candidate in payload["locked_candidates"]:
            explanations.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "reason": "Concept khớp các tín hiệu đã ghi nhận trong hồ sơ.",
                    "why_it_fits": candidate["matched_signals"] or ["nằm trong ngân sách"],
                    "personalization_tip": "Thêm thiệp viết tay mang dấu ấn cá nhân.",
                    "verify_before_buying": candidate["possible_risks"],
                    "budget_note": "Giá chỉ là ước tính và cần kiểm tra lại.",
                    # Các trường giả này phải bị Tool 4 bỏ qua.
                    "rank": 999,
                    "estimated_price_range_vnd": {"minimum": 1, "maximum": 1},
                }
            )
        return {"explanations": explanations}

    raise RuntimeError("Fake LLM không nhận diện được prompt.")


def _run_smoke_tests() -> None:
    """Chạy smoke tests deterministic, không gọi API thật."""
    tests: list[tuple[str, Callable[[], bool]]] = []

    def add(name: str, check: Callable[[], bool]) -> None:
        tests.append((name, check))

    configure_tool_llm(_fake_structured_llm)
    description = (
        "Tặng quà sinh nhật cho bạn thân 21 tuổi, hướng nội, thích đọc sách "
        "và trà, thích quà ý nghĩa, không thích mùi hương, ngân sách 800.000 VND."
    )
    step1 = json.loads(extract_recipient_profile(description))
    profile = step1.get("recipient_profile", {})
    step2 = json.loads(analyze_recipient_profile(profile))
    analysis = step2.get("profile_analysis", {})
    step3 = json.loads(generate_gift_candidates(profile, analysis, max_candidates=5))
    candidates = step3.get("ranked_candidates", [])
    step4 = json.loads(explain_recommendations(profile, analysis, candidates, top_k=5))

    add("Tool 1 trả JSON success", lambda: step1.get("ok") is True)
    add("Tool 2 trả JSON success", lambda: step2.get("ok") is True)
    add("Tool 3 sinh candidate", lambda: step3.get("ok") is True and bool(candidates))
    add("Tool 4 sinh explanation", lambda: step4.get("ok") is True)
    add("Description rỗng trả error", lambda: json.loads(extract_recipient_profile(""))["ok"] is False)
    add("Input sai kiểu trả error", lambda: json.loads(extract_recipient_profile(123))["ok"] is False)  # type: ignore[arg-type]
    add("Exclusion thắng interest", lambda: "mùi hương" in profile.get("exclusions", []))
    add("Candidate vượt budget bị loại", lambda: any(item["reason_code"] == "CANDIDATE_OVER_BUDGET" for item in step3["generation_summary"]["rejected_candidates"]))
    add("Candidate vi phạm exclusion bị loại", lambda: any(item["reason_code"] == "CANDIDATE_VIOLATES_EXCLUSION" for item in step3["generation_summary"]["rejected_candidates"]))
    add("Python tự gán candidate_id", lambda: all(re.fullmatch(r"C\d{3}", item["candidate_id"]) for item in candidates))
    add("Candidate có score", lambda: all(isinstance(item.get("score"), int) for item in candidates))
    add("Candidate có score_breakdown", lambda: all(isinstance(item.get("score_breakdown"), dict) for item in candidates))
    add("Rank bắt đầu từ 1 liên tục", lambda: [item["rank"] for item in candidates] == list(range(1, len(candidates) + 1)))
    add("Score giảm dần", lambda: [item["score"] for item in candidates] == sorted((item["score"] for item in candidates), reverse=True))
    add("Tool 4 giữ nguyên ID", lambda: [item["candidate_id"] for item in step4["recommendations"]] == [item["candidate_id"] for item in candidates])
    add("Tool 4 giữ nguyên rank", lambda: [item["rank"] for item in step4["recommendations"]] == [item["rank"] for item in candidates])
    add("Tool 4 khóa khoảng giá", lambda: [item["estimated_price_range_vnd"] for item in step4["recommendations"]] == [item["estimated_price_range_vnd"] for item in candidates])
    add("top_k âm trả error", lambda: json.loads(explain_recommendations(profile, analysis, candidates, top_k=-1))["ok"] is False)
    add("Thiếu budget trả error ở Tool 3", lambda: json.loads(generate_gift_candidates({**profile, "budget_vnd": None}, analysis))["error"]["code"] == "MISSING_BUDGET")
    add("Mọi output parse được", lambda: all(isinstance(item, dict) for item in (step1, step2, step3, step4)))

    configure_tool_llm(None)
    add("LLM chưa cấu hình trả error", lambda: json.loads(extract_recipient_profile(description))["error"]["code"] == "TOOL_LLM_NOT_CONFIGURED")

    failures = 0
    print("🧪 Role 2 — ReAct Generate Tools smoke tests\n")
    for name, check in tests:
        try:
            passed = bool(check())
        except Exception as exc:  # Test harness only.
            passed = False
            detail = str(exc)
        else:
            detail = ""
        if passed:
            print(f"[PASS] {name}")
        else:
            failures += 1
            print(f"[FAIL] {name}: {detail}")

    print(f"\nKết quả: {len(tests) - failures}/{len(tests)} PASS")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_smoke_tests()
