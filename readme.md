To run this program

First, ensure Ollama is installed and running on your system.

open your terminal and pull the three required vision models:

```
ollama pull gemma3:4b
ollama pull llava
ollama pull moondream
```

make sure to make a venv and activate it 

python3 -m venv venv
source venv/bin/activate

run the following command in the venv to install the dependancies
pip install -r requirements.txt

While in the venv, run the command

python3 app.py 

http://127.0.0.1:8080 in your browser to use the app.

