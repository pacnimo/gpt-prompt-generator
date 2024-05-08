import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.2")

def format_prompt(message, job_profession):
    system_prompt = f"You are a People Generator. You generate profiles based on user input. Please ensure that all responses are written in the second-person perspective. For example, a response should be structured as follows: 'You are (Name), (Age) years old from (Country), and an expert in the field of {job_profession}.' This format must be strictly followed in all outputs."
    prompt = f"<s>[SYS] {system_prompt} [/SYS][INST] {message} [/INST]</s>"
    return prompt

def generate(message, job_profession, temperature=0.9, max_new_tokens=4192, top_p=0.95, repetition_penalty=1.0):
    # Parse the input to determine job profession based on the presence of a message.
    actual_job = message if message else job_profession

    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "do_sample": True,
        "seed": 42,
    }

    formatted_prompt = format_prompt("", actual_job)  # message should be empty here if we're using job from input
    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output

css = """
#mkd {
    height: 500px; 
    overflow: auto; 
    border: 1px solid #ccc; 
}
"""

with gr.Blocks(css=css) as gpt:
    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML("<h1>Settings</h1>")
            job_dropdown = gr.Dropdown(
                label="Choose a job profession",
                choices=["Relationship Expert", "Sales Manager", "Blue Team Hacker", "Senior Web Developer", "C++ Developer", "Article Author", "News Anchor", "Finance Advisor"],
                value="Relationship Expert"  # Initial value
            )
        with gr.Column(scale=3):
            gr.HTML("<h1><center>GPT Prompt Generator<h1><center>")
            message_input = gr.Textbox(label="Input your message (overrides job dropdown)", placeholder="Type a job profession here to override the dropdown...")
            generate_button = gr.Button("Generate")
            output_area = gr.Textbox(label="AI Response", interactive=False, lines=10)
            generate_button.click(
                fn=generate,
                inputs=[message_input, job_dropdown],
                outputs=output_area
            )

    gr.Markdown("""
    ---
    ### Meta Information
    **Project Title**: GPT Prompt Generator
    **Github**: [https://github.com/pacnimo/](https://github.com/pacnimo/)
    **Description**: GPT Prompt Generator is Free and Easy to Use. Create your GPT Prompt Based on the Profession.
    **Footer**: © 2024 by [GitHub](https://github.com/pacnimo/). All rights reserved.
    """) # Meta, project description, and footer added here

gpt.launch(debug=True)

