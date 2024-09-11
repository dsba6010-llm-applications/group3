import requests

def test_health():
    url = "https://sam060584--vllm-openai-compatible-serve.modal.run/health"  
    response = requests.get(url)

    # Verify status code
    assert response.status_code == 200 

    # Verify content-type
    assert response.headers["Content-Type"] == "application/json"; 
