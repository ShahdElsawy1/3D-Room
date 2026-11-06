from conftest import signup


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_signup_login_me(client):
    h = signup(client, "User@Example.com")
    me = client.get("/auth/me", headers=h)
    assert me.status_code == 200 and me.json()["email"] == "user@example.com"   # normalised to lower case

    r = client.post("/auth/login", json={"email": "user@example.com", "password": "password123"})
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer" and body["expires_in"] > 0 and body["user"]["email"] == "user@example.com"
    assert "password_hash" not in body["user"]


def test_swagger_form_login(client):
    signup(client, "f@example.com")
    r = client.post("/auth/token", data={"username": "f@example.com", "password": "password123"})
    assert r.status_code == 200 and r.json()["access_token"]


def test_duplicate_and_bad_credentials(client):
    signup(client, "d@example.com")
    assert client.post("/auth/signup", json={"email": "d@example.com", "password": "password123"}).status_code == 409
    assert client.post("/auth/login", json={"email": "d@example.com", "password": "wrongpass1"}).status_code == 401
    assert client.post("/auth/login", json={"email": "nobody@example.com", "password": "password123"}).status_code == 401
    assert client.post("/auth/signup", json={"email": "x@example.com", "password": "short"}).status_code == 422
    assert client.post("/auth/signup", json={"email": "not-an-email", "password": "password123"}).status_code == 422


def test_protected_routes_need_valid_token(client):
    assert client.get("/projects").status_code == 401
    assert client.get("/projects", headers={"Authorization": "Bearer garbage"}).status_code == 401
    assert client.get("/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_token_signed_with_other_secret_rejected(client):
    from app.security import create_access_token
    forged = create_access_token("someone", "another-secret-0123456789-abcdefghijkl", 60)
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
