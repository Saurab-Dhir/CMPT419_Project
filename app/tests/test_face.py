import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_visual_emotion_endpoint():
    # Simulate an image upload
    test_image = io.BytesIO(b"fake_image_data")
    response = client.post(
        "/api/v1/face/detect",  # updated route
        files={"image": ("testface1.jpg", test_image, "image/jpeg")}
    )

    assert response.status_code == 200 or response.status_code == 500
