import os
import gradio as gr
import fire
from enum import Enum
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from transformers import TextIteratorStreamer
from llama_chat_format import format_to_llama_chat_style

# class syntax
class Model_Type(Enum):
    gptq = 1
    full_precision = 2

def get_model_type(model_name):
  if "gptq" in model_name.lower():
    return Model_Type.gptq
  else:
    return Model_Type.full_precision

def initialize_gpu_model_and_tokenizer(model_name, model_type):
    if model_type == Model_Type.gptq:
      model = AutoGPTQForCausalLM.from_quantized(model_name, device_map="auto", use_safetensors=True, use_triton=False)
      tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
      model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=True)
      tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    return model, tokenizer

def init_auto_model_and_tokenizer(model_name, model_type, file_name=None):
  model_type = get_model_type(model_name)
  model, tokenizer = initialize_gpu_model_and_tokenizer(model_name, model_type=model_type)
  return model, tokenizer

def run_ui(model, tokenizer, is_chat_model, model_type):
  with gr.Blocks() as demo:
      chatbot = gr.Chatbot()
      msg = gr.Textbox()
      clear = gr.Button("Clear")

      def user(user_message, history):
          return "", history + [[user_message, None]]

      def bot(history):
          if is_chat_model:
              instruction = format_to_llama_chat_style(history)
          else:
              instruction =  history[-1][0]

          history[-1][1] = ""
          kwargs = dict(temperature=0.6, top_p=0.9)

          streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, Timeout=5)
          inputs = tokenizer(instruction, return_tensors="pt").to(model.device)
          kwargs["max_new_tokens"] = 512
          kwargs["input_ids"] = inputs["input_ids"]
          kwargs["streamer"] = streamer
          thread = Thread(target=model.generate, kwargs=kwargs)
          thread.start()

          for token in streamer:
              history[-1][1] += token
              yield history

      msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
      clear.click(lambda: None, None, chatbot, queue=False)
  demo.queue()
  demo.launch(share=True, debug=True)

def main(model_name=None, file_name=None):
    assert model_name is not None, "model_name argument is missing."

    is_chat_model = 'chat' in model_name.lower()
    model_type = get_model_type(model_name)

    model, tokenizer = init_auto_model_and_tokenizer(model_name, model_type, file_name)
    run_ui(model, tokenizer, is_chat_model, model_type)

if __name__ == '__main__':
  fire.Fire(main)