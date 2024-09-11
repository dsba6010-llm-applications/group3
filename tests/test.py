import requests, pytest

def test_health():
    url = "https://sam060584--vllm-openai-compatible-serve.modal.run/health"  
    response = requests.get(url, verify=False)

    # Verify status code
    assert response.status_code == 200 

def test_models():
    url = "https://sam060584--vllm-openai-compatible-serve.modal.run/v1/models"  
    response = requests.get(url, verify=False)

    # Verify status code
    assert response.status_code == 200 
    assert response.headers["Content-Type"] == "application/json; charset=UTF-8"

def test_version():
    url = "https://sam060584--vllm-openai-compatible-serve.modal.run/version"  
    response = requests.get(url, verify=False)

    # Verify status code
    assert response.status_code == 200 
    assert response.headers["Content-Type"] == "application/json; charset=UTF-8"

@pytest.fixture
def body_data():
    return {
        "model": "/models/NousResearch/Meta-Llama-3-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }

def test_chatCompletion(body_data):
    url = "https://sam060584--vllm-openai-compatible-serve.modal.run/v1/chat/completions"  


    headers = {"Content-Type": "application/json; charset=utf-8", "Authorization": "Bearer go-niners"}
    
    response = requests.post(url, headers=headers, json=body_data, verify=False)

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/json; charset=UTF-8"

    
    data = response.json()
    assert "msg" in data 
    
    




   
