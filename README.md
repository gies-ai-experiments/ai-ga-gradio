# Canvas Discussion Grader With Feedback 🎓📝

This application is designed to automate the grading process of Canvas discussions. It uses advanced AI models to grade student responses based on a provided rubric, and also provides developmental feedback to students. The application is built using the Gradio SDK.

## Features 🚀

- **Automated Grading**: The application grades Canvas discussions based on a provided rubric, saving educators valuable time.
- **Developmental Feedback**: In addition to grading, the application provides feedback to students, helping them understand how they can improve.
- **Easy to Use**: Simply provide the URL of the Canvas discussion and your Canvas API Key, and the application will handle the rest.

## How to Use 🛠️

1. **Ingest Canvas Discussions**: Enter your Canvas Discussion URL and your Canvas API Key, then click the 'Ingest' button. The application will download the discussion data and prepare it for grading.

2. **Start Grading**: Once the data has been ingested, click the 'Grade' button to start the grading process. The application will grade each student's response based on the provided rubric.

3. **Download Results**: After grading is complete, you can download the results as a CSV file. This file contains the grades for each student, along with the feedback provided by the application.

4. **Ask Questions**: You can ask questions about the grading process or the results. The application uses an AI model to provide answers to your questions.

## Requirements 📋

This application requires the following Python packages:

- [gradio](file:///Users/rohanmarwaha/IdeaProjects/ai-ga-gradio/requirements.txt#8%2C11-8%2C11)
- [openai](file:///Users/rohanmarwaha/IdeaProjects/ai-ga-gradio/requirements.txt#13%2C9-13%2C9)
- [tiktoken](file:///Users/rohanmarwaha/IdeaProjects/ai-ga-gradio/requirements.txt#265%2C9-265%2C9)
- [faiss-cpu](file:///Users/rohanmarwaha/IdeaProjects/ai-ga-gradio/requirements.txt#62%2C1-62%2C1)
- [bs4](file:///Users/rohanmarwaha/IdeaProjects/ai-ga-gradio/ingest.py#8%2C6-8%2C6)
- [pathvalidate](file:///Users/rohanmarwaha/IdeaProjects/ai-ga-gradio/grader.py#15%2C6-15%2C6)
- [unstructured](file:///Users/rohanmarwaha/IdeaProjects/ai-ga-gradio/requirements.txt#42%2C11-42%2C11)

You can install these packages using pip:
```bash
pip install -r requirements.txt
```


## Running the Application 🏃‍♀️

To run the application, simply execute the `app.py` script:
```bash
python app.py
```


This will launch the Gradio interface, where you can interact with the application on http://localhost:7860.

## Contributing 🤝

Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

## License 📄

This project is licensed under the MIT License.