import sys
from types import ModuleType


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
_install_stub_qdrant_client()
_install_stub_httpx()
