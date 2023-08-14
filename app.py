import asyncio
import glob
import os
import shutil
import time
import traceback

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

import utils
from csv_agent import CSVAgent
from grader import Grader
from grader_qa import GraderQA
from ingest import ingest_canvas_discussions
from utils import reset_folder

load_dotenv()

pickle_file = "vector_stores/canvas-discussions.pkl"
index_file = "vector_stores/canvas-discussions.index"

grading_model = 'gpt-4'
qa_model = 'gpt-4'

llm = ChatOpenAI(model_name=qa_model, temperature=0, verbose=True)
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

grader = None
grader_qa = None
disabled = gr.update(interactive=False)
enabled = gr.update(interactive=True)


def add_text(history, text):
    print("Question asked: " + text)
    response = run_model(text)
    history = history + [(text, response)]
    print(history)
    return history, ""


def run_model(text):
    global grader, grader_qa
    start_time = time.time()
    print("start time:" + str(start_time))
    try:
        response = grader_qa.agent.run(text)
    except Exception as e:
        response = "I need a break. Please ask me again in a few minutes"
        print(traceback.format_exc())

    sources = []
    # for document in response['source_documents']:
    #     sources.append(str(document.metadata))

    source = ','.join(set(sources))
    # response = response['answer'] + '\nSources: ' + str(len(sources))
    end_time = time.time()
    # # If response contains string `SOURCES:`, then add a \n before `SOURCES`
    # if "SOURCES:" in response:
    #     response = response.replace("SOURCES:", "\nSOURCES:")
    response = response + "\n\n" + "Time taken: " + str(end_time - start_time)
    print(response)
    print(sources)
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
    return history, disabled, disabled, disabled, enabled


def start_grading(history):
    global grader, grader_qa
    text = f"Start grading discussions from {url}"
    if grader:
        # if grader.llm.model_name != grading_model:
        #     grader = Grader(grading_model)
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
    return history, disabled, enabled, enabled, enabled


def start_downloading():
    # files = glob.glob("output/*.csv")
    # if files:
    #     file = files[0]
    #     return gr.outputs.File(file)
    # else:
    #     return "File not found"
    print(grader.csv)
    return grader.csv, gr.update(visible=True), gr.update(value=process_csv_text(), visible=True)

def get_headers():
    df = process_csv_text()
    return list(df.columns)
    

def get_first_message(history):
    global grader_qa
    history = [(None,
                'Get feedback on your canvas discussions. Add your discussion url and get your discussions graded in instantly.')]
    return get_grading_status(history)


def get_grading_status(history):
    global grader, grader_qa
    # Check if grading is complete
    if os.path.isdir('output') and len(glob.glob("output/*.csv")) > 0 and len(glob.glob("docs/*.json")) > 0 and len(
            glob.glob("docs/*.html")) > 0:
        if not grader:
            grader = Grader(qa_model)
            grader_qa = GraderQA(grader, embeddings)
        elif not grader_qa:
            grader_qa = GraderQA(grader, embeddings)
        if len(history) == 1:
            history = history + [(None, 'Grading is already complete. You can now ask questions')]
        enable_fields(False, False, False, False, True, True, True)
    # Check if data is ingested
    elif len(glob.glob("docs/*.json")) > 0 and len(glob.glob("docs/*.html")):
        if not grader_qa:
            grader = Grader(qa_model)
        if len(history) == 1:
            history = history + [(None, 'Canvas data is already ingested. You can grade discussions now')]
        enable_fields(False, False, False, True, True, False, False)
    else:
        history = history + [(None, 'Please ingest data and start grading')]
        enable_fields(True, True, True, False, False, False, False)
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


def reset_data():
    # Use shutil.rmtree() to delete output, docs, and vector_stores folders, reset grader and grader_qa, and get_grading_status, reset and return history
    global grader, grader_qa
    #If there's data in docs/output folder during grading [During Grading]
    if os.path.isdir('output') and len(glob.glob("output/*.csv")) > 0 and len(glob.glob("docs/*.json")) > 0 and len(
            glob.glob("docs/*.html")) > 0:
        reset_folder('output')
        reset_folder('docs')
        grader = None
        grader_qa = None
        history = [(None, 'Data reset successfully')]
        return history, enabled, enabled, enabled, disabled, disabled, disabled
    # If there's data in docs folder [During Ingestion]
    elif len(glob.glob("docs/*.json")) > 0 and len(glob.glob("docs/*.html")):
        reset_folder('docs')
        history = [(None, 'Data reset successfully')]
        return history, enabled, enabled, enabled, disabled, disabled, disabled
    #If there's data in vector_stores folder
    elif len(glob.glob("vector_stores/*.faiss")) > 0 or len(glob.glob("vector_stores/*.pkl")) > 0:
        reset_folder('vector_stores')
        history = [(None, 'Data reset successfully')]
        return history, enabled, enabled, enabled, disabled, disabled, disabled


def get_output_dir(orig_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output', orig_name)
    return output_dir


def upload_grading_results(file, history):
    global grader, grader_qa
    # Delete output folder and save the file in output folder
    if os.path.isdir('output'):
        shutil.rmtree('output')
    os.mkdir('output')
    if os.path.isdir('vector_stores'):
        shutil.rmtree('vector_stores')
    os.mkdir('vector_stores')
    # get current path
    path = os.path.join("output", os.path.basename(file.name))
    # Copy the uploaded file from its temporary location to the desired location
    shutil.copyfile(file.name, path)
    grader_qa = CSVAgent(llm, embeddings, path)
    history = [(None, 'Grading results uploaded successfully. Start Chatting!')]
    return history


def bot(history):
    return history

def process_csv_text():
    file_path = utils.get_csv_file_name()
    df = pd.read_csv(file_path)
    return df

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

        submit = gr.Button(value="Step 1: Submit", variant="secondary", )
    with gr.Row():
        table = gr.Dataframe(label ='Canvas CSV Output', type="pandas", overflow_row_behaviour="paginate", visible = False, wrap=True)

    with gr.Row():
        grade = gr.Button(value="Step 2: Grade", variant="secondary")
        download = gr.Button(value="Step 3: View Grading Output", variant="secondary")
        file = gr.components.File(label="CSV Output", container=False, visible=False).style(height=100)
        reset = gr.ClearButton(value="Reset")

    chatbot = gr.Chatbot([], label="Chat with grading results", elem_id="chatbot", height=400)

    with gr.Row():
        with gr.Column(scale=3):
            txt = gr.Textbox(
                label="Ask questions about how students did on the discussion",
                placeholder="Enter text and press enter, or upload an image", lines=1
            )
        ask = gr.Button(value="Step 4: Chat", variant="secondary", scale=1)
        upload = gr.UploadButton(label="Upload grading results", type="file", file_types=["csv"], scale=0.5)

    chatbot.value = get_first_message([])
    
    with gr.Row():
        table = gr.Dataframe(label ='Canvas CSV Output', type="pandas", overflow_row_behaviour="paginate", visible = False, wrap=True)
    
    submit.click(ingest, inputs=[url, canvas_api_key, chatbot], outputs=[chatbot, url, canvas_api_key, submit, grade],
                 postprocess=False).then(
        bot, chatbot, chatbot
    )

    grade.click(start_grading, inputs=[chatbot], outputs=[chatbot, grade, download, txt, ask],
                postprocess=False).then(
        bot, chatbot, chatbot
    )

    download.click(start_downloading, inputs=[], outputs=[file, file, table]).then(
        bot, chatbot, chatbot
    )

    txt.submit(add_text, [chatbot, txt], [chatbot, txt], postprocess=False).then(
        bot, chatbot, chatbot
    )

    ask.click(add_text, inputs=[chatbot, txt], outputs=[chatbot, txt], postprocess=False, ).then(
        bot, chatbot, chatbot
    )

    reset.click(reset_data, inputs=[], outputs=[chatbot, url, canvas_api_key, submit, table, grade, download]).success(
        bot, chatbot, chatbot)

    upload.upload(upload_grading_results, inputs=[upload, chatbot], outputs=[chatbot], postprocess=False, ).then(
        bot, chatbot, chatbot)

if __name__ == "__main__":
    demo.queue()
    demo.queue(concurrency_count=5)
    demo.launch(debug=True, )
