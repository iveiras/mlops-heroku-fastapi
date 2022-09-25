from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


# Write tests using the same syntax as with the requests module.
def test_api_without_query():
    r = client.get("/items/12")
    assert r.status_code == 200
    assert r.json() == {"fetch": "Fetched 1 of 12"}


def test_api_with_query():
    r = client.get("/items/12?count=100")
    assert r.status_code == 200
    assert r.json() == {"fetch": "Fetched 100 of 12"}


def test_api_wrong_url():
    r = client.get("/100?count=23")
    assert r.status_code != 200


if __name__ == "__main__":
    test_api_with_query()
    test_api_without_query()
    test_api_wrong_url()
