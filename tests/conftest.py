"""Shared helpers: run the app's api_client against an in-process FastAPI app."""
from app import api_client


def route_api_client(client, monkeypatch) -> None:
    """Send every api_client call to ``client`` (a FastAPI TestClient) instead of the network."""
    def fake_request(method, base_url, path, **kwargs):
        r = client.request(method, path, **kwargs)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = None
            raise api_client.APIError(f"HTTP {r.status_code}", status=r.status_code,
                                      detail=detail if isinstance(detail, str) else None)
        return r

    monkeypatch.setattr(api_client, "_request", fake_request)
