from fastapi.testclient import TestClient


def test_health():
    from serve.api import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_scenes():
    from serve.api import app
    client = TestClient(app)
    resp = client.get("/scenes")
    assert resp.status_code == 200
    assert "defect_3d" in resp.json()["scenes"]


def test_get_inspections_empty():
    from serve.api import app
    client = TestClient(app)
    resp = client.get("/inspections")
    assert resp.status_code == 200
