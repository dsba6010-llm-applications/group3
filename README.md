# 🦙 Llama3-8b-instruct Chatbot  
__Group 3__: Eric (product mgr.), Yaxin (data/prompt engr.), Lakshmi (app dev.), Gaurav (LLM ops.)  
  
## ⚙️ Setup  
1. Make sure you have signed up for a [Modal account](https://modal.com/).
  
2. Clone the repo like this:

```bash
git clone --depth 1 https://github.com/dsba6010-llm-applications/group3.git
```

> [!WARNING]
> Our virtual environment was accidentally included in the initial push. It has since been removed from the repo but will be present in git history.
> Be sure to include ```--depth 1``` when cloning the repo to exclude git history and avoid downloading the virtual environment.

3. Then `cd` into the folder `group3`.

4. Create a virtual environment, activate it, and install dependencies.

```python
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> [!TIP]
> If you're using Windows CMD, the 2nd line will be `.\venv\Scripts\activate.bat`. 
> Alternatively, if you're using Windows PowerShell, it would be `.\venv\Scripts\activate.ps1`

5. Setup Modal locally.

```python
python -m modal setup
```

A browser window will open and you should select your Modal account. 

You should receive a `Web authentication finished successfully!` message.  
  
## 🍽️ Serving the Llama3-8b-instruct on Modal

You will need to set a secret token (as a mock OPEN AI API key) to authenticate as a [Modal Secret](https://modal.com/docs/guide/secrets).

Go to Modal→Your account→Dashboard→Secrets and select creating a Custom Secret. At step two, put `DSBA_LLAMA3_KEY` under `Key` and the "OpenAI API key" under `Value`. Click next. It will ask you to give your secret a name. Name your secret `dsba-llama3-key`.

> [!NOTE]
> You will need to create an `.env` file with `DBSA_LLAMA3_KEY=<your secret value>`. 

After supplying the secret in Modal, you should be able to run the following command with no error:

```bash
modal deploy backend/src/api.py
```

This will then provide you a URL endpoint: https://`your-workspace-name`--vllm-openai-compatible-serve.modal.run> 

You can view the Swagger API doc at<https://`your-workspace-name`--vllm-openai-compatible-serve.modal.run/docs  
  
## 🏃🏻‍♀️ Running inference using client.py

>[!IMPORTANT]
>Make sure you have a `.env` file with your token such that: `DSBA_LLAMA3_KEY=<secret-token>`

Now, you can run:

```
$ python backend/src/client.py
🧠: Looking up available models on server at https://your-workspace-name--vllm-openai-compatible-serve.modal.run/v1/. This may trigger a boot!
🧠: Requesting completion from model /models/NousResearch/Meta-Llama-3-8B-Instruct
👉: You are a poetic assistant, skilled in writing satirical doggerel with creative flair.
👤: Compose a limerick about baboons and racoons.
🤖: There once were two creatures quite fine,
Baboons and raccoons, a curious combine,
They raided the trash cans with glee,
In the moon's silver shine,
Together they dined, a messy entwine.
```

## 🤖 Streamlit Chatbot
>[!IMPORTANT]
>First, create the `frontend/.streamlit/secrets.toml` directory and file such that:
>```toml
>DSBA_LLAMA3_KEY="<your key>"
>MODAL_BASE_URL="https://<your url>--vllm-openai-compatible-serve.modal.run"  
>```
>This will use your LLM serving endpoint created above. Do not include 'v1/' in the URL.

### To run locally:

```bash
$ python -m streamlit run frontend/app.py
```

### To run on Modal:

See "Serving the Llama3-8b-instruct on Modal" section to create the LLM serving endpoint on Modal.

#### You can run a temporary "dev" environment to test:

```bash
# to test
$ modal serve frontend/modal/serve_streamlit.py
```

#### Or deploy it as a new app to Modal:

```bash
# when ready to deploy
$ modal deploy frontend/modal/serve_streamlit.py
```
