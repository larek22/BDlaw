from app.id_utils import chunk_sha, point_id_from_sha


def test_chunk_sha_deterministic():
    assert chunk_sha("hello") == chunk_sha("hello")
    assert chunk_sha("hello") != chunk_sha("world")


def test_point_id_from_sha_is_uuid():
    sha = chunk_sha("sample")
    uuid_str = point_id_from_sha(sha)
    assert uuid_str.count("-") == 4
    assert point_id_from_sha(sha) == uuid_str
