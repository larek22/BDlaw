import sys
from types import ModuleType
from typing import Any, Dict


def _install_stub_openai() -> None:
    if "openai" in sys.modules:
        return

    module = ModuleType("openai")

    class _StubEmbeddings:
        def create(self, *args, **kwargs):  # pragma: no cover - tests monkeypatch real client
            raise NotImplementedError("Stub OpenAI embeddings should not be called during tests")

    class _StubCompletions:
        def create(self, *args, **kwargs):  # pragma: no cover - tests monkeypatch real client
            raise NotImplementedError("Stub OpenAI chat completions should not be called during tests")

    class _StubChat:
        def __init__(self) -> None:
            self.completions = _StubCompletions()

    class _StubOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            self.embeddings = _StubEmbeddings()
            self.chat = _StubChat()

    class _StubOpenAIError(Exception):
        def __init__(self, *args, **kwargs):  # pragma: no cover - used for compatibility only
            message = args[0] if args else kwargs.get("message", "")
            super().__init__(message)

    module.OpenAI = _StubOpenAI
    module.OpenAIError = _StubOpenAIError
    module.PermissionDeniedError = _StubOpenAIError
    sys.modules["openai"] = module


def _install_stub_pydantic() -> None:
    if "pydantic" in sys.modules:
        return

    module = ModuleType("pydantic")

    _UNSET = object()

    class ValidationError(Exception):
        pass

    class _FieldInfo:
        def __init__(self, default: Any = _UNSET, default_factory=None) -> None:
            self.default = default
            self.default_factory = default_factory

    def Field(*, default: Any = _UNSET, default_factory=None):
        return _FieldInfo(default=default, default_factory=default_factory)

    class BaseModel:
        _field_info: Dict[str, _FieldInfo] = {}

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            fields: Dict[str, _FieldInfo] = {}
            annotations = getattr(cls, "__annotations__", {})
            for name in annotations:
                default = cls.__dict__.get(name, _UNSET)
                if isinstance(default, _FieldInfo):
                    fields[name] = default
                    setattr(cls, name, _UNSET)
                elif default is not _UNSET:
                    fields[name] = _FieldInfo(default=default)
                else:
                    fields[name] = _FieldInfo()
            cls._field_info = fields

        def __init__(self, **data: Any) -> None:
            for name, info in self._field_info.items():
                if name in data:
                    value = data[name]
                else:
                    if info.default is not _UNSET:
                        value = info.default
                    elif info.default_factory is not None:
                        value = info.default_factory()
                    else:
                        value = None
                setattr(self, name, value)

        @classmethod
        def model_validate(cls, data: Any):
            if isinstance(data, cls):
                return data
            if not isinstance(data, dict):
                raise ValidationError("Expected mapping for model validation")
            return cls(**data)

        def model_dump(self) -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            for name in self._field_info:
                value = getattr(self, name)
                if isinstance(value, BaseModel):
                    result[name] = value.model_dump()
                elif isinstance(value, list):
                    result[name] = [
                        item.model_dump() if isinstance(item, BaseModel) else item
                        for item in value
                    ]
                else:
                    result[name] = value
            return result

    module.BaseModel = BaseModel
    module.Field = Field
    module.ValidationError = ValidationError
    sys.modules["pydantic"] = module


def _install_stub_qdrant_client() -> None:
    if "qdrant_client" in sys.modules:
        return

    module = ModuleType("qdrant_client")
    module.__version__ = "0.0-test"

    class _StubQdrantClient:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
            pass

        def recreate_collection(self, *args, **kwargs):  # pragma: no cover
            return None

        def get_collection(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

        def upsert(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

        def count(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

        def search(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

        def query_points(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

        def scroll(self, *args, **kwargs):  # pragma: no cover
            return ([], None)

        def create_payload_index(self, *args, **kwargs):  # pragma: no cover
            return None

        def delete(self, *args, **kwargs):  # pragma: no cover
            return None

        def create_snapshot(self, *args, **kwargs):  # pragma: no cover
            return type("Snapshot", (), {"name": "stub.snapshot"})()

        def download_snapshot(self, *args, **kwargs):  # pragma: no cover
            return None

        def upload_snapshot(self, *args, **kwargs):  # pragma: no cover
            return None

    module.QdrantClient = _StubQdrantClient

    http_module = ModuleType("qdrant_client.http")
    models_module = ModuleType("qdrant_client.http.models")

    class VectorParams:
        def __init__(self, size: int, distance):
            self.size = size
            self.distance = distance

    class VectorParamsMap(dict):
        pass

    class Distance:
        COSINE = "COSINE"

    class PointStruct:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload

    class FieldCondition:
        def __init__(self, key: str, match):
            self.key = key
            self.match = match

    class MatchValue:
        def __init__(self, value):
            self.value = value

    class MatchText:
        def __init__(self, text):
            self.text = text

    class Filter:
        def __init__(self, must=None, should=None):
            self.must = must or []
            self.should = should or []

    class FilterSelector:
        def __init__(self, filter: Filter):
            self.filter = filter

    class PayloadSchemaType:
        KEYWORD = "keyword"
        TEXT = "text"

    class MatchAny:
        def __init__(self, any):
            self.any = any

    class HnswConfigDiff:
        def __init__(self, *args, **kwargs):
            pass

    class OptimizersConfigDiff:
        def __init__(self, *args, **kwargs):
            pass

    models_module.VectorParams = VectorParams
    models_module.VectorParamsMap = VectorParamsMap
    models_module.Distance = Distance
    models_module.PointStruct = PointStruct
    models_module.FieldCondition = FieldCondition
    models_module.MatchValue = MatchValue
    models_module.MatchText = MatchText
    models_module.Filter = Filter
    models_module.FilterSelector = FilterSelector
    models_module.PayloadSchemaType = PayloadSchemaType
    models_module.MatchAny = MatchAny
    models_module.HnswConfigDiff = HnswConfigDiff
    models_module.OptimizersConfigDiff = OptimizersConfigDiff

    http_module.models = models_module
    module.http = http_module

    sys.modules["qdrant_client"] = module
    sys.modules["qdrant_client.http"] = http_module
    sys.modules["qdrant_client.http.models"] = models_module


def _install_stub_httpx() -> None:
    if "httpx" in sys.modules:
        return

    module = ModuleType("httpx")

    class _Response:
        def __init__(self) -> None:
            self._json = {}
            self.text = ""
            self.headers = {"content-type": "application/json"}

        def json(self):
            return self._json

        def raise_for_status(self):  # pragma: no cover - always succeeds
            return None

    def get(*args, **kwargs):  # pragma: no cover - deterministic stub
        return _Response()

    module.get = get
    module.Response = _Response
    sys.modules["httpx"] = module


_install_stub_openai()
_install_stub_pydantic()
_install_stub_qdrant_client()
_install_stub_httpx()
