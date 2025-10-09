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

    module.QdrantClient = _StubQdrantClient

    http_module = ModuleType("qdrant_client.http")
    models_module = ModuleType("qdrant_client.http.models")

    class VectorParams:
        def __init__(self, size: int, distance):
            self.size = size
            self.distance = distance

    class Distance:
        COSINE = "COSINE"

    class PointStruct:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload

    models_module.VectorParams = VectorParams
    models_module.Distance = Distance
    models_module.PointStruct = PointStruct

    http_module.models = models_module
    module.http = http_module

    sys.modules["qdrant_client"] = module
    sys.modules["qdrant_client.http"] = http_module
    sys.modules["qdrant_client.http.models"] = models_module


_install_stub_openai()
_install_stub_qdrant_client()
