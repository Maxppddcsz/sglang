from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint
import time

@function
def text_qa(s, question):
    s += "Q: " + question + "\n"
    st = time.time()
    s += "A:" + gen("answer", stop="\n")
    ed = time.time()
    print(f"Latency is {ed-st} s")

set_default_backend(RuntimeEndpoint("http://localhost:30000"))

states = text_qa.run_batch(
    [
        {"question": "What is the capital of the United Kingdom?"},
        {"question": "What is the capital of France?"},
        {"question": "What is the capital of Japan?"},
    ],
    progress_bar=True
)