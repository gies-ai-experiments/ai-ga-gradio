import asyncio
import os
import time
import glob
import gradio as gr
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from grader import Grader
from ingest import ingest_canvas_discussions
from utils import GraderQA

load_dotenv()

pickle_file = "vector_stores/canvas-discussions.pkl"
index_file = "vector_stores/canvas-discussions.index"

grading_model = 'gpt-4'
qa_model = 'gpt-3.5-turbo-16k'

llm = ChatOpenAI(model_name=qa_model, temperature=0, verbose=True)
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

grader = None
grader_qa = None


def add_text(history, text):
    print("Question asked: " + text)
    get_grading_status(history)
    response = run_model(text)
    history = history + [(text, response)]
    print(history)
    return history, ""


def run_model(text):
    global grader, grader_qa
    start_time = time.time()
    print("start time:" + str(start_time))
    response = grader_qa.chain(text)
    sources = []
    for document in response['source_documents']:
        sources.append(str(document.metadata))
        print(sources)

    source = ','.join(set(sources))
    response = response['answer'] + '\nSources: ' + source
    end_time = time.time()
    # # If response contains string `SOURCES:`, then add a \n before `SOURCES`
    # if "SOURCES:" in response:
    #     response = response.replace("SOURCES:", "\nSOURCES:")
    response = response + "\n\n" + "Time taken: " + str(end_time - start_time)
    print(response)
    print("Time taken: " + str(end_time - start_time))
    return response


def set_model(history):
    history = get_first_message(history)
    return history

def ingest(url, canvas_api_key, history):
    global grader, llm, embeddings
    text = f"Downloaded discussion data from {url} to start grading"
    ingest_canvas_discussions(url, canvas_api_key)
    grader = Grader(grading_model)
    response = "Ingested canvas data successfully"
    history = history + [(text, response)]
    return get_grading_status(history)

def start_grading(url, canvas_api_key, history):
    global grader, grader_qa
    text = f"Start grading discussions from {url}"
    if not url or not canvas_api_key:
        response = "Please enter all the fields to initiate grading"
    elif grader:
        if grader.llm.model_name != grading_model:
            grader = Grader(grading_model)
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Use the event loop to run the async function
            loop.run_until_complete(grader.run_chain())
            grader_qa = GraderQA(grader, embeddings)
            response = "Grading done"
        finally:
            # Close the loop after use
            loop.close()
    else:
        response = "Please ingest data before grading"
    history = history + [(text, response)]
    return history


def start_downloading():
    files = glob.glob("output/*.csv")
    if files:
        file = files[0]
        return gr.outputs.File(file)
    else:
        return "File not found"


def get_first_message(history):
    global grader_qa
    history = [(None,
                'Get feedback on your canvas discussions. Add your discussion url and get your discussions graded in instantly.')]
    history = get_grading_status(history)
    return history


def get_grading_status(history):
    global grader, grader_qa
    # Check if grading is complete
    if os.path.isdir('output') and len(glob.glob("docs/*.json")) > 0 and len(glob.glob("docs/*.html")) > 0:
        if not grader:
            grader = Grader(qa_model)
            grader_qa = GraderQA(grader, embeddings)
        elif not grader_qa:
            grader_qa = GraderQA(grader, embeddings)
        history = history + [(None, 'Grading is already complete. You can now ask questions')]
        enable_fields(False, False, False, False, True, True, True)
    # Check if data is ingested
    elif len(glob.glob("docs/*.json")) > 0 and len(glob.glob("docs/*.html")):
        if not grader_qa:
            grader = Grader(qa_model)
        history = history + [(None, 'Canvas data is already ingested. You can grade discussions now')]
        enable_fields(False, False, False, True, True, False, False)
    else:
        history = history + [(None, 'Please ingest data and start grading')]
        url.disabled = True
        enable_fields(True, True, True, True, True, False, False)
    return history


# handle enable/disable of fields
def enable_fields(url_status, canvas_api_key_status, submit_status, grade_status,
                  download_status, chatbot_txt_status, chatbot_btn_status):
    url.interactive = url_status
    canvas_api_key.interactive = canvas_api_key_status
    submit.interactive = submit_status
    grade.interactive = grade_status
    download.interactive = download_status
    txt.interactive = chatbot_txt_status
    ask.interactive = chatbot_btn_status
    if not chatbot_txt_status:
        txt.placeholder = "Please grade discussions first"
    else:
        txt.placeholder = "Ask a question"
    if not url_status:
        url.placeholder = "Data already ingested"
    if not canvas_api_key_status:
        canvas_api_key.placeholder = "Data already ingested"


def bot(history):
    return history


with gr.Blocks() as demo:
    gr.Markdown(f"<h2><center>{'Canvas Discussion Grading With Feedback'}</center></h2>")

    with gr.Row():
        url = gr.Textbox(
            label="Canvas Discussion URL",
            placeholder="Enter your Canvas Discussion URL"
        )

        canvas_api_key = gr.Textbox(
            label="Canvas API Key",
            placeholder="Enter your Canvas API Key", type="password"
        )

    with gr.Row():
        submit = gr.Button(value="Submit", variant="secondary", )
        grade = gr.Button(value="Grade", variant="secondary")
        download = gr.Button(value="Download", variant="secondary")
        reset = gr.Button(value="Reset", variant="secondary")

    chatbot = gr.Chatbot([], label="Chat with grading results", elem_id="chatbot", height=400)

    with gr.Row():
        with gr.Column(scale=3):
            txt = gr.Textbox(
                label="Ask questions about how students did on the discussion",
                placeholder="Enter text and press enter, or upload an image", lines=1
            )
        ask = gr.Button(value="Ask", variant="secondary", scale=1)

    chatbot.value = get_first_message([])
    submit.click(ingest, inputs=[url, canvas_api_key, chatbot], outputs=[chatbot],
                 postprocess=False).then(
        bot, chatbot, chatbot
    )

    grade.click(start_grading, inputs=[url, canvas_api_key, chatbot], outputs=[chatbot],
                postprocess=False).then(
        bot, chatbot, chatbot
    )

    download.click(start_downloading, inputs=[], outputs=[chatbot], postprocess=False).then(
        bot, chatbot, chatbot
    )

    txt.submit(add_text, [chatbot, txt], [chatbot, txt], postprocess=False).then(
        bot, chatbot, chatbot
    )

    ask.click(add_text, inputs=[chatbot, txt], outputs=[chatbot, txt], postprocess=False,).then(
        bot, chatbot, chatbot
    )

    set_model(chatbot)

if __name__ == "__main__":
    demo.queue()
    demo.queue(concurrency_count=5)
    demo.launch(debug=True, )
