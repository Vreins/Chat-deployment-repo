import panel as pn
import gradio as gr
from bokeh.embed import server_document
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
app= FastAPI()
templates= Jinja2Templates(directory="templates")
CUSTOM_PATH="/gradio"
# CAll pn server
@app.get("/")
async def bkapp_page(request: Request):
    script=server_document('http://127.0.0.1:5000/app')
    return templates.TemplateResponse("base.html", {"request":request, "script":script})
