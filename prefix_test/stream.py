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

states = text_qa.run(
    question="What is the capital of France?",
    temperature=0.1,
    stream=True
)

for out in states.text_iter():
    print(out, end="", flush=True)

states = text_qa.run(
    question="What is the capital of France?",
    temperature=0.1,
    stream=True
)

for out in states.text_iter():
    print(out, end="", flush=True)